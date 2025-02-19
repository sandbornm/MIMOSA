import sys
import logging
from os.path import join
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchsummary import summary

from sklearn.model_selection import RepeatedKFold

from ray import tune

from comet_ml import Experiment  # importing after torch disables autologging

import util

from . import datasets
from . import models
from . import metrics


def run(args):
	if args.mode.lower() == 'train':
		train(vars(args))
	elif args.mode.lower() == 'cv' or args.mode.lower() == 'cross_val':
		cross_val(vars(args))
	elif args.mode.lower() == 'predict':
		predict(vars(args))
	elif args.mode.lower() == 'strat':
		strat(vars(args))
	else:
		logging.error('Unknown mode entered: %s' % args.mode)


def create_config(args):
	"""
	Create a config from a dictionary of args
	"""
	examples_dir = args['examples_dir']
	labels_csv = args['labels_csv']
	name = args['name']
	load = args['load']
	frequency = args['frequency']
	mode = args['mode']
	modality = args['modality']
	arch = args['arch']
	size = args['size']
	epochs = args['epochs']
	batchsize = args['batchsize']
	lr = args['lr']
	val = args['val']
	pretrain = args['pretrain']
	variant = args['variant']
	hidden = args['hidden']
	optim = args['optim']
	loss = args['loss']
	n_classes = args['n_classes']

	# handle transforms
	if args['modality'] == 'image':
		args['transform'] = Compose([ToTensor()])
	else:
		args['transform'] = None

	# handle size arg per modality
	if 'bytes' in modality.lower():
		args['size'] = np.prod(args['size'])

	# create dataset
	dataset = datasets.build_dataset(args)
	n_examples = dataset.n_examples

	# build net
	net = models.build_model(args, n_classes)

	if loss.lower() == 'bce':
		criterion = nn.BCELoss()
	elif loss.lower() == 'cce':
		criterion = nn.CrossEntropyLoss()
	else:
		raise ValueError('Unknown loss function: ', loss)

	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda")

	input_sz = dataset[0]['example'].shape
	logging.info(f'Dataset: # of examples = {n_examples}, # of classes = {n_classes}, input size = {input_sz}')

	# handle device memory errors
	oom = False
	try:
		summary(net, input_sz)
	except RuntimeError:
		oom = True
		torch.cuda.empty_cache()

	if oom:
		device = torch.device('cpu')
		summary(net, input_sz, device=device)
	elif torch.cuda.device_count() > 1:
		net = nn.DataParallel(net)

	logging.info(f'Using device {device}')

	# attempt load if spec'd
	if mode.lower() == 'predict' or mode.lower() == 'test':
		if not load:
			raise AssertionError('must specify a model to load in predict and test modes')

	if load:
		net.load_state_dict(
			torch.load(load, map_location=device)
		)
		logging.info('net loaded from %s' % load)

	net.to(device=device)

	if 'sgd' in optim.lower():
		optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	elif 'adam' in optim.lower():
		optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	elif 'rmsprop' in optim.lower():
		optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
	else:
		raise ValueError('Unsupported optimizer: ', optim)

	config = {'net': net,
	          'dataset': dataset,
	          'device': device,
	          'epochs': epochs,
	          'batch_size': batchsize,
	          'lr': lr,
	          'val_percent': val,
	          'frequency': frequency,
	          'criterion': criterion,
	          'exp_name': name,
	          'optim': optimizer
	          }
	return config


# https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
def train_epoch(net, device, dataloader, loss_fn, optimizer, classes=None, **kwargs):
	n_train = kwargs['n_train']
	epoch = kwargs['epoch']
	epochs = kwargs['epochs']

	train_loss = 0.0
	train_metrics = None
	net.train()
	with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
		for batch in dataloader:
			examples = batch['example']
			labels = batch['label']
			labels = torch.squeeze(labels).float()

			optimizer.zero_grad()
			oom = False
			try:
				examples, labels, net = examples.to(device), labels.to(device), net.to(device)
				output = net(examples)
			except RuntimeError:
				oom = True
				torch.cuda.empty_cache()

			if oom:
				examples, labels, net = examples.cpu(), labels.cpu(), net.cpu()
				output = net(examples)

			output = output.squeeze()
			loss = loss_fn(output, labels)
			loss.backward()
			optimizer.step()

			output = output.detach().cpu()
			labels = labels.detach().cpu().numpy()
			loss = loss.detach().cpu()

			train_loss += loss.item() * examples.size(0)

			if classes:
				mtr = metrics.report(output.detach().cpu(), labels.detach().cpu().numpy(), classes)
				train_metrics = mtr if not train_metrics else metrics.merge_reports(train_metrics, mtr)
			else:
				mtr = metrics.calculate_metrics(output, labels)
				train_metrics = mtr if not train_metrics else metrics.merge_metrics(train_metrics, mtr)

			pbar.update(examples.shape[0])

	return train_loss, train_metrics


def val_epoch(net, device, dataloader, loss_fn, classes=None):
	val_loss = 0.0
	val_metrics = None
	net.eval()
	for batch in dataloader:
		examples = batch['example']
		labels = batch['label']
		labels = torch.squeeze(labels).float()

		with torch.no_grad():
			oom = False
			try:
				examples, labels, net = examples.to(device), labels.to(device), net.to(device)
				output = net(examples)
			except RuntimeError:
				oom = True
				torch.cuda.empty_cache()

			if oom:
				examples, labels, net = examples.cpu(), labels.cpu(), net.cpu()
				output = net(examples)

			output = output.squeeze()
			loss = loss_fn(output, labels)

		output = output.detach().cpu()
		labels = labels.detach().cpu().numpy()
		loss = loss.detach().cpu()

		val_loss += loss.item() * examples.size(0)

		if classes:
			mtr = metrics.report(output.detach().cpu(), labels.detach().cpu().numpy(), classes)
			val_metrics = mtr if not val_metrics else metrics.merge_reports(val_metrics, mtr)
		else:
			mtr = metrics.calculate_metrics(output, labels)
			val_metrics = mtr if not val_metrics else metrics.merge_metrics(val_metrics, mtr)

	return val_loss, val_metrics


def cross_val(args, tuning=False):
	"""
	Repeated K-Fold cross-validation for a PyTorch classifier
	"""
	config = create_config(args)

	net = config['net']
	dataset = config['dataset']
	device = config['device']
	epochs = config['epochs']
	batch_size = config['batch_size']
	lr = config['lr']
	criterion = config['criterion']
	exp_name = config['exp_name']
	optimizer = config['optim']

	# comet setup
	experiment = Experiment(
		api_key="k86kE4n1wy7wQkkCmvZeFAV3M",
		project_name="mimosa",
		workspace="zstoebs",
	)
	if tuning:
		experiment.add_tag(exp_name)  # tag with initial tuning id
		ID = '_'.join([str(args[key]) for key in args.keys()])
		h = hash(ID)
		h += sys.maxsize if h < 0 else 0
		exp_name += '_' + str(h)

	experiment.set_name(exp_name)
	experiment.add_tag(args['arch'])

	experiment.log_parameters(args)

	classes = dataset.classes

	foldperf = {}
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	for fold, (train_idx, val_idx) in enumerate(cv.split(np.arange(len(dataset)))):

		logging.info('Fold {}'.format(fold + 1))

		train_sampler = SubsetRandomSampler(train_idx)
		val_sampler = SubsetRandomSampler(val_idx)
		train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
		val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

		model = copy.deepcopy(net)
		model.to(device)

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

		history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_ranking': [],
		           'val_ranking': []}

		for epoch in range(epochs):
			train_kwargs = {'n_train': len(train_sampler), 'epoch': epoch, 'epochs': epochs}
			train_loss, train_metrics = train_epoch(net, device, train_loader, criterion, optimizer,
			                                        **train_kwargs)
			val_loss, val_metrics = val_epoch(net, device, val_loader, criterion)

			train_loss /= len(train_sampler)
			val_loss /= len(val_sampler)

			metrics.log_metrics(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)
			# log_reports(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)

			mean_train_acc = np.mean(train_metrics['samples/f1'])
			mean_val_acc = np.mean(val_metrics['samples/f1'])
			mean_train_ranking = np.mean(train_metrics['ranking'])
			mean_val_ranking = np.mean(val_metrics['ranking'])

			if mean_val_ranking > 0.9:
				experiment.add_tag('>0.9')
			if mean_val_ranking > 0.95:
				experiment.add_tag('>0.95')

			scheduler.step(mean_val_acc)

			logging.info(
				"Epoch:{}/{} \n "
				"AVG Training Ranking:{:.3f} \n"
				"AVG Val Ranking:{:.3f} \n"
				"AVG Training Loss:{:.3f} \n "
				"AVG Val Loss:{:.3f} \n "
				"AVG Training Acc {:.2f} \n "
				"AVG Val Acc {:.2f}".format(
					epoch + 1,
					epochs,
					mean_train_ranking,
					mean_val_ranking,
					train_loss,
					val_loss,
					mean_train_acc,
					mean_val_acc))

			history['train_loss'].append(train_loss)
			history['val_loss'].append(val_loss)
			history['train_acc'].append(mean_train_acc)
			history['val_acc'].append(mean_val_acc)
			history['train_ranking'].append(mean_train_ranking)
			history['val_ranking'].append(mean_val_ranking)

			if tuning:
				with tune.checkpoint_dir(epoch) as checkpoint_dir:
					path = join(checkpoint_dir, "checkpoint")
					torch.save((net.state_dict(), optimizer.state_dict()), path)

				tune.report(loss=val_loss, accuracy=mean_val_acc, ranking=mean_val_ranking)

		foldperf['fold{}'.format(fold + 1)] = history

	torch.cuda.empty_cache()


def train(args, tuning=False):
	"""
	Standard iterative training for a PyTorch classifier
	"""
	config = create_config(args)

	net = config['net']
	dataset = config['dataset']
	device = config['device']
	epochs = config['epochs']
	batch_size = config['batch_size']
	lr = config['lr']
	val_percent = config['val_percent']
	frequency = config['frequency']
	criterion = config['criterion']
	exp_name = config['exp_name']
	optimizer = config['optim']

	# comet setup
	experiment = Experiment(
		api_key="k86kE4n1wy7wQkkCmvZeFAV3M",
		project_name="mimosa",
		workspace="zstoebs",
	)
	if tuning:
		experiment.add_tag(exp_name)  # tag with initial tuning id
		ID = '_'.join([str(args[key]) for key in args.keys()])
		h = hash(ID)
		h += sys.maxsize if h < 0 else 0
		exp_name += '_' + str(h)

	experiment.set_name(exp_name)
	experiment.add_tag(args['arch'])
	experiment.log_parameters(args)

	util.makedir(join(args['cp_dir'], 'cp'))
	save_dir = join(args['cp_dir'], 'cp', exp_name)

	net.train()

	n_val = int(len(dataset) * val_percent)
	n_train = len(dataset) - n_val
	train, val = random_split(dataset, [n_train, n_val])
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
	val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False,
	                        drop_last=True)

	logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

	classes = dataset.classes

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

	history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_ranking': [], 'val_ranking': []}

	for epoch in range(epochs):
		train_kwargs = {'n_train': n_train, 'epoch': epoch, 'epochs': epochs}
		train_loss, train_metrics = train_epoch(net, device, train_loader, criterion, optimizer, **train_kwargs)
		val_loss, val_metrics = val_epoch(net, device, val_loader, criterion)

		train_loss /= n_train
		val_loss /= n_val

		metrics.log_metrics(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)
		# log_reports(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)

		mean_train_acc = np.mean(train_metrics['samples/f1'])
		mean_val_acc = np.mean(val_metrics['samples/f1'])
		mean_train_ranking = np.mean(train_metrics['ranking'])
		mean_val_ranking = np.mean(val_metrics['ranking'])

		if mean_val_ranking > 0.9:
			experiment.add_tag('>0.9')
		if mean_val_ranking > 0.95:
			experiment.add_tag('>0.95')

		scheduler.step(mean_val_acc)

		logging.info(
			"Epoch:{}/{} \n "
			"AVG Training Ranking:{:.3f} \n"
			"AVG Val Ranking:{:.3f} \n"
			"AVG Training Loss:{:.3f} \n "
			"AVG Val Loss:{:.3f} \n "
			"AVG Training Acc {:.2f} \n "
			"AVG Val Acc {:.2f}".format(
				epoch + 1,
				epochs,
				mean_train_ranking,
				mean_val_ranking,
				train_loss,
				val_loss,
				mean_train_acc,
				mean_val_acc))

		history['train_loss'].append(train_loss)
		history['val_loss'].append(val_loss)
		history['train_acc'].append(mean_train_acc)
		history['val_acc'].append(mean_val_acc)
		history['train_ranking'].append(mean_train_ranking)
		history['val_ranking'].append(mean_val_ranking)

		if tuning:
			with tune.checkpoint_dir(epoch) as checkpoint_dir:
				path = join(checkpoint_dir, "checkpoint")
				torch.save((net.state_dict(), optimizer.state_dict()), path)

			tune.report(loss=val_loss, accuracy=mean_val_acc, ranking=mean_val_ranking)

		# save checkpoint
		if frequency and epoch % frequency == 0:
			util.makedir(save_dir)
			torch.save(net.state_dict(),
			           join(save_dir, f'{exp_name}_epoch{epoch + 1}.pth'))
			logging.info(f'Checkpoint {epoch + 1} saved !')

	# save final model
	util.makedir(save_dir)
	torch.save(net.state_dict(),
	           join(save_dir, f'{exp_name}_final.pth'))
	logging.info(f'Final model saved !')

	torch.cuda.empty_cache()


def predict(args):
	"""
	Standard iterative prediction for a PyTorch classifier
	"""
	config = create_config(args)

	net = config['net']
	dataset = config['dataset']
	device = config['device']
	batch_size = config['batch_size']
	exp_name = config['exp_name']

	# comet setup
	experiment = Experiment(
		api_key="k86kE4n1wy7wQkkCmvZeFAV3M",
		project_name="mimosa",
		workspace="zstoebs",
	)

	experiment.set_name(exp_name)
	experiment.add_tag(args['arch'])
	experiment.log_parameters(args)

	net.eval()

	n_examples = len(dataset)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)

	logging.info(f'''Starting prediction:
        Batch size:      {batch_size}
        Number of examples:   {n_examples}
        Device:          {device.type}
    ''')

	classes = dataset.classes

	preds = None
	index = []
	with tqdm(total=n_examples, desc=f'Predicting {exp_name}', unit='img') as pbar:
		for batch in loader:
			examples = batch['example']
			hashes = batch['hash']

			with torch.no_grad():
				oom = False
				try:
					examples, net = examples.to(device), net.to(device)
					output = net(examples)
				except RuntimeError:
					oom = True
					torch.cuda.empty_cache()

				if oom:
					examples, net = examples.cpu(), net.cpu()
					output = net(examples)

				output = output.squeeze()

			output = output.detach().cpu()

			preds = output if preds is None else np.vstack([preds, output])
			index += hashes

			pbar.update(examples.shape[0])

	df = pd.DataFrame(data=preds, index=index, columns=classes)
	df.to_csv(join('results', exp_name + '.csv'))

	experiment.log_dataframe_profile(df)
	experiment.log_table(join('results', exp_name + '.csv'))

	torch.cuda.empty_cache()


def strat(args):
	"""
	*TODO*
	Stratified analysis of training with different dataset sizes
	"""

	config = create_config(args)

	net = config['net']
	dataset = config['dataset']
	device = config['device']
	epochs = config['epochs']
	batch_size = config['batch_size']
	lr = config['lr']
	val_percent = config['val_percent']
	frequency = config['frequency']
	criterion = config['criterion']
	exp_name = config['exp_name']
	optimizer = config['optim']

	for step, per in enumerate(range(10, 101, 10)):
		frac = per // 100

		net.reinit()

		# comet setup
		experiment = Experiment(
			api_key="k86kE4n1wy7wQkkCmvZeFAV3M",
			project_name="mimosa",
			workspace="zstoebs",
		)

		experiment.set_name('strat_' + exp_name + '_step=%.1f' % frac)
		experiment.add_tag(args['arch'])
		experiment.log_parameters(args)

		net.train()

		n_val = int(len(dataset) * val_percent)
		n_train = len(dataset) - n_val
		train, val = random_split(dataset, [n_train, n_val])
		train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
		val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False,
		                        drop_last=True)

		logging.info(f'''Starting training:
	            Epochs:          {epochs}
	            Batch size:      {batch_size}
	            Learning rate:   {lr}
	            Training size:   {n_train}
	            Validation size: {n_val}
	            Device:          {device.type}
	        ''')

		classes = dataset.classes

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

		history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_ranking': [], 'val_ranking': []}

		for epoch in range(epochs):
			train_kwargs = {'n_train': n_train, 'epoch': epoch, 'epochs': epochs}
			train_loss, train_metrics = train_epoch(net, device, train_loader, criterion, optimizer, **train_kwargs)
			val_loss, val_metrics = val_epoch(net, device, val_loader, criterion)

			train_loss /= n_train
			val_loss /= n_val

			metrics.log_metrics(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)
			# log_reports(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)

			mean_train_acc = np.mean(train_metrics['samples/f1'])
			mean_val_acc = np.mean(val_metrics['samples/f1'])
			mean_train_ranking = np.mean(train_metrics['ranking'])
			mean_val_ranking = np.mean(val_metrics['ranking'])

			if mean_val_ranking > 0.9:
				experiment.add_tag('>0.9')
			if mean_val_ranking > 0.95:
				experiment.add_tag('>0.95')

			scheduler.step(mean_val_acc)

			logging.info(
				"Epoch:{}/{} \n "
				"AVG Training Ranking:{:.3f} \n"
				"AVG Val Ranking:{:.3f} \n"
				"AVG Training Loss:{:.3f} \n "
				"AVG Val Loss:{:.3f} \n "
				"AVG Training Acc {:.2f} \n "
				"AVG Val Acc {:.2f}".format(
					epoch + 1,
					epochs,
					mean_train_ranking,
					mean_val_ranking,
					train_loss,
					val_loss,
					mean_train_acc,
					mean_val_acc))

			history['train_loss'].append(train_loss)
			history['val_loss'].append(val_loss)
			history['train_acc'].append(mean_train_acc)
			history['val_acc'].append(mean_val_acc)
			history['train_ranking'].append(mean_train_ranking)
			history['val_ranking'].append(mean_val_ranking)

		torch.cuda.empty_cache()
