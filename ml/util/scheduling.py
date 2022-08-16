"""
Scheduling samples to configs based on output of the ML multilabel
classifier. 

Key considerations: 
- load balancing

** If it fails, update probs / labels and rerun scheduling algo
to find where sample goes next.
"""

from os.path import join
import argparse
import pandas as pd
import numpy as np
from numpy.random import choice
from itertools import compress
from random import sample

config_names = ['hopper1_kvm_patched_conf1',
                'hopper1_kvm_patched_conf2',
                'hopper2_vmware_conf3',
                'hopper4_vmware_conf2',
                'hopper5_vmware_conf2_vmtools',
                'hopper6_kvm_legacy_conf1',
                'hopper6_kvm_legacy_conf2',
                'hopper3_vbox_conf1',
                'hopper3_vbox_conf2',
                'hopper8_vbox_conf1_guestadditions',
                'hopper8_vbox_conf2_guestadditions',
                'hopper7_qemu_legacy_conf1',
                'hopper7_qemu_legacy_conf2']

config_costs = {                # startup, runtime, resources
	'hopper1_kvm_patched_conf1': [536.4, 180.8, 35000],
	'hopper1_kvm_patched_conf2': [536.4, 180.8, 25000],
	'hopper2_vmware_conf3': [477, 60.7, 10000],
	'hopper4_vmware_conf2': [477, 644.6, 4000],
	'hopper5_vmware_conf2_vmtools': [477, 73.5, 39000],
	'hopper7_qemu_legacy_conf1': [538.1, 327.4, 52000],
	'hopper7_qemu_legacy_conf2': [538.1, 327.4, 42000],
	'hopper8_vbox_conf1_guestadditions': [422.7, 142, 13666],
	'hopper8_vbox_conf2_guestadditions': [422.7, 142, 12345],
	'hopper3_vbox_conf1': [422.7, 73.4, 54321],
	'hopper3_vbox_conf2': [304.7, 73.4, 22222],
	'hopper6_kvm_legacy_conf1': [543.3, 327.4, 12321],
	'hopper6_kvm_legacy_conf2': [543.3, 327.4, 99999]
}

out_dir = '.'


def get_args():
	parser = argparse.ArgumentParser(description='Scheduling args')

	parser.add_argument('--p_csv', '-p', type=str, required=True, help='path to CSV containing model output')
	parser.add_argument('--n_servers', '-ns', type=int, default=13, help='number of servers')
	parser.add_argument('--strategy', '-st', type=str, default='wrr', help='strategy for allocating samples to configs. [wrr (weighted round robin) | random | koth]')
	parser.add_argument('--success_rate', '-sr', type=int, default=100, choices=range(0, 101), metavar='[0-100]', help='percent success rate for simulation. Ideal = 100.')
	parser.add_argument('--out_dir', '-o', type=str, default='.', help='Output directory')

	return parser.parse_args()


def get_data(csv):
	"""
	Read and preprocess model output data to prepare for scheduling
	"""
	df = pd.read_csv(csv)
	cols = df.columns
	hashes = df[cols[0]]
	probs = df[cols[1:]].to_numpy()

	return hashes, probs


def threshold(arr, thresh=0.5):
	"""
	Binary threshold an array around a given value
	"""
	return np.where(arr < thresh, 0, 1)


def find_ind_in_schedule(ind, schedule):
	"""
	Scours schedule for ind and returns info
	"""
	for server, info in schedule.items():
		for j, load in enumerate(info['loads']):
			inds = load['inds']
			if ind in inds:
				pos = inds.index(ind)
				return server, info['configs'][j], load['probs'][pos]

	raise ValueError('Index %d not found in schedule' % ind)


def WRR_alloc(samples, configs):
	"""
	Weighted round robin allocation
	"""

	# BALANCER AFTER SCHEDULING DOES A BETTER JOB AT LOAD BALANCING THAN SOFT LIMIT
	# round robin allocation
	# C = len(samples) // len(configs)  # soft limit per config
	# print('Soft limit per config: ', C)

	negatives = []
	loads = {config: {'inds': [], 'probs': []} for config in configs}
	for ind, sample in samples.items():
		s_probs = sample['probs']
		s_labels = sample['labels']

		# record negative prediction sample
		if sum(s_labels) == 0:
			negatives += [ind]

		# weighted allocation
		else:
			# filter each samples predicted configs and probs of success
			pos_confs = list(compress(configs, s_labels))
			pos_probs = list(compress(s_probs, s_labels))

			# sort based on prob (descending) and save to dict
			sorted_info = sorted(zip(pos_confs, pos_probs), key=lambda a: a[1], reverse=True)

			allocated = False
			# for conf, prob in sorted_info:
			#     if not allocated and len(loads[conf]) < C:
			#         loads[conf]['inds'] += [ind]
			#         loads[conf]['probs'] += [prob]
			#         allocated = True

			# if sample couldn't be allocated, just allocate to likeliest config
			if not allocated:
				mx_conf, mx_prob = sorted_info[0]
				loads[mx_conf]['inds'] += [ind]
				loads[mx_conf]['probs'] += [mx_prob]

	return loads, negatives


def random_alloc(samples, configs):
	"""
	Random allocation
	"""
	n_configs = len(configs)
	negatives = []
	loads = {config: {'inds': [], 'probs': []} for config in configs}
	for ind, sample in samples.items():
		s_probs = sample['probs']
		s_labels = sample['labels']

		# assign randomly regardless
		j = choice(n_configs)
		rand_conf = configs[j]

		# record negative prediction sample
		if s_labels[j] == 0 or sum(s_labels) == 0:
			negatives += [ind]

		loads[rand_conf]['inds'] += [ind]
		loads[rand_conf]['probs'] += [s_probs[j]]

	return loads, negatives


def koth_alloc(samples, configs):
	"""
	KotH (aka assign all to best config) allocation
	"""
	# find best config
	labels = []
	for _, sample in samples.items():
		s_labels = sample['labels']
		labels += [s_labels]

	labels = np.vstack(labels)
	sums = np.sum(labels, axis=0)
	best_ind = np.argmax(sums)
	best_conf = configs[best_ind]
	print('**Best conf: ', best_conf)

	negatives = []
	loads = {config: {'inds': [], 'probs': []} for config in configs}
	for ind, sample in samples.items():
		s_probs = sample['probs']
		s_labels = sample['labels']

		# record negative prediction sample and doesn't run on koth
		if s_labels[best_ind] == 0 or sum(s_labels) == 0:
			negatives += [ind]

		# assign to supposed best conf
		loads[best_conf]['inds'] += [ind]
		loads[best_conf]['probs'] += [s_probs[best_ind]]

	return loads, negatives


def allocator(samples, configs, strategy='wrr'):

	strat = strategy.lower()
	if strat == 'wrr':
		loads, negatives = WRR_alloc(samples, configs)
	elif strat == 'random':
		loads, negatives = random_alloc(samples, configs)
	elif strat == 'koth':
		loads, negatives = koth_alloc(samples, configs)
	else:
		raise ValueError('Unknown strategy: %s' % strategy)

	return loads, negatives


def compute_cost(config, costs, n_samples):
	startup, runtime, resources = costs[config]
	return startup + n_samples * runtime, resources


def balancer(schedule, costs):
	"""
	Balance loads across available servers in schedule
	"""

	for server in schedule:

		# if server is empty
		if schedule[server]['runtime'] == 0:

			# find the longest server
			longest_server = max(schedule, key=lambda server: schedule[server]['runtime'])
			print('Balancing from server %d to server %d' % (longest_server, server))

			# if multiple loads, move longest load to empty server
			if len(schedule[longest_server]['costs']) > 1:
				mx_ind = max(enumerate(schedule[longest_server]['costs']), key=lambda x: x[1][0])
				config = schedule[longest_server]['configs'][mx_ind]
				load = schedule[longest_server]['loads'][mx_ind]
				cost = schedule[longest_server]['costs'][mx_ind]
				load_runtime = cost[0]

				# assign
				schedule[server]['configs'] += [config]
				schedule[server]['loads'] += [load]
				schedule[server]['costs'] += [cost]
				schedule[server]['runtime'] += load_runtime

				# update
				del schedule[longest_server]['configs'][mx_ind]
				del schedule[longest_server]['loads'][mx_ind]
				del schedule[longest_server]['costs'][mx_ind]
				schedule[longest_server]['runtime'] -= load_runtime

			# if only one load, split in half and move to empty server
			else:
				config = schedule[longest_server]['configs'][0]
				load = schedule[longest_server]['loads'][0]
				inds = load['inds']
				probs = load['probs']

				n_samples = len(inds)
				if n_samples > 1:
					half = n_samples // 2

					first_inds = inds[:half]
					second_inds = inds[half:]

					first_probs = probs[:half]
					second_probs = probs[half:]

					first_load = {'inds': first_inds, 'probs': first_probs}
					second_load = {'inds': second_inds, 'probs': second_probs}

					first_cost = compute_cost(config, costs, len(first_inds))
					second_cost = compute_cost(config, costs, len(second_inds))

					# assign
					schedule[server]['configs'] += [config]
					schedule[server]['loads'] += [second_load]
					schedule[server]['costs'] += [second_cost]
					schedule[server]['runtime'] += second_cost[0]

					# update
					schedule[longest_server]['loads'][0] = first_load
					schedule[longest_server]['costs'][0] = first_cost
					schedule[longest_server]['runtime'] -= second_cost[0]


def scheduler(probs, configs, costs, n_servers, strategy='wrr', thresh=0.5):
	"""
	Schedule samples to configs using N servers based on costs
	and model output probs, with load balancing maximizing
	probability of sample execution

	Job status: 0 = pending, 1 = scheduled

	Parameters:
		probs = numpy array of output probabilities. Rows: samples, Columns: sample's prob on config
		Configs = list of names that correspond to column order in prob array
		Costs = dict of costs associated with each config name
	Return:
		schedule = array / graphic of jobs running on each config
		negatives = inds of samples that were not predicted to run on any config
	"""
	schedule = {server: {'configs': [], 'loads': [], 'costs': [], 'runtime': 0} for server in range(n_servers)}

	# build sample info sheet
	labels = threshold(probs, thresh)
	samples = {ind: {'probs': prob, 'labels': label} for ind, (prob, label) in enumerate(zip(probs, labels))}

	# sort columns based on prob (ascending) then flip for descending
	# sorted_inds = np.flipud(np.argsort(probs, axis=0))

	### PART 1: ALLOCATION
	print('Allocating...')

	loads, negatives = allocator(samples, configs, strategy)
	print('# of negatives: ', len(negatives))

	### PART 2: SCHEDULING
	print('Scheduling to servers...')

	# compute costs for each load as in runtime and resources as a product each sample in load
	# treating loads as static units
	# assumption: sample domain randomly selected --> extremely unlikely to predict only one config

	total_samples = 0
	for config, load in loads.items():
		s_inds = load['inds']

		n_samples = len(s_inds)
		total_samples += n_samples
		print('# of samples allocated to %s: ' % config, n_samples)
		if n_samples == 0:
			continue

		load_runtime, resources = compute_cost(config, costs, n_samples)
		print('>>> with total runtime: %d' % load_runtime)
		print('>>> with resources: %d' % resources)

		# greedy assignment to server based on current run time
		server = min(schedule, key=lambda server: schedule[server]['runtime'])

		print('>>> assigned to server %d' % server)

		schedule[server]['configs'] += [config]
		schedule[server]['loads'] += [load]
		schedule[server]['costs'] += [(load_runtime, resources)]
		schedule[server]['runtime'] += load_runtime

	# load balancing
	if total_samples >= n_servers:
		balancer(schedule, costs)

	return schedule, negatives


def simulater(probs, configs, costs, n_servers, strategy, success=1.0, thresh=0.5):
	"""
	Simulate a run of scheduling based on servers, strategy, and success rate
	"""
	print('** Strategy is %s' % strategy)
	print('** Success rate is %.1f' % success)

	rem = list(range(len(probs)))
	results = {}
	k = 0
	data = probs.copy()
	while rem:
		k += 1

		print('\n\n** Iteration: ', k)
		schedule, negatives = scheduler(data[rem, :], configs, costs, n_servers, strategy, thresh)

		# empty schedule bc all ended up as negatives --> done
		if len(negatives) == len(rem):
			results[k] = {'predictive': 0, 'runtime': 0, 'negatives': negatives}
			break

		# compute total runtime (aka longest running server)
		longest_server = max(schedule, key=lambda server: schedule[server]['runtime'])
		total_runtime = schedule[longest_server]['runtime']
		print('Server %s runs the longest with runtime %d' % (longest_server, total_runtime))

		# compute average success prob total and on each server
		avgs = []
		lengths = []
		for server in schedule:
			loads = schedule[server]['loads']
			if not schedule[server]['configs']:
				continue

			s = 0
			l = 0
			for load in loads:
				l_probs = load['probs']
				s += sum(l_probs)
				l += len(l_probs)

			avg = s / l
			print('Average predictive probability for server %d is %.2f' % (server, avg))

			avgs += [avg]
			lengths += [l]

		num = sum([avg * l for avg, l in zip(avgs, lengths)])
		denom = sum(lengths)
		mean = num / denom
		print('Overall predictive probability on schedule is %.2f' % mean)

		# store mean and runtime for this iter in results
		results[k] = {'predictive': mean, 'runtime': total_runtime, 'negatives': negatives}

		# update rem indices based on negatives
		neg_rem = [rem[ind] for ind in negatives]
		new_rem = [ind for ind in rem if ind not in neg_rem]

		# update rem indices based on success rate
		subsample = int(len(new_rem) * (1.0 - success))
		new_rem = sample(new_rem, subsample)

		# update probs based on new rem s.t. failed config = 0.0
		for ind in new_rem:
			old = rem.index(ind)
			_, config, _ = find_ind_in_schedule(old, schedule)
			j = configs.index(config)
			data[ind, j] = 0.0

		rem = new_rem

	print('\n\n** # of iters: ', len(results))
	return results


def server_vs_time(probs, configs, costs, max_servers, success):
	"""
	Compute scalability data for # of servers vs time
	"""
	strategies = ['ideal', 'random', 'koth', 'mimosa']
	overall = {strategy: {'n_servers': [], 'predictive': [], 'runtime': [], 'iters': [], 'coverage': []} for strategy in strategies}
	for strategy in strategies:
		for n_servers in range(1, max_servers+1):
			if strategy == 'mimosa':
				results = simulater(probs, configs, costs, n_servers, 'wrr', success)
			elif strategy == 'ideal':
				results = simulater(probs, configs, costs, n_servers, 'wrr', success=1.0)
			else:
				results = simulater(probs, configs, costs, n_servers, strategy, thresh=0.7)

			preds = []
			runtimes = []
			n_negatives = 0
			for k, result in results.items():
				p = result['predictive']
				rt = result['runtime']
				n_negs = len(result['negatives'])

				preds += [p]
				runtimes += [rt] if rt else []
				n_negatives += n_negs

			pred = sum(preds) / len(preds)
			runtime = sum(runtimes)
			coverage = 1 - n_negatives / len(probs)
			print('>>> Average predictive value is: ', pred)
			print('>>> Average runtime is: ', runtime)
			print('>>> Coverage is: ', coverage)

			overall[strategy]['n_servers'] += [n_servers]
			overall[strategy]['runtime'] += [runtime]
			overall[strategy]['iters'] += [len(results)]
			overall[strategy]['predictive'] += [pred]
			overall[strategy]['coverage'] += [coverage]

	fname = join(out_dir, 'servers=%d_vs_time_success=%.2f.xlsx' % (max_servers, success))
	with pd.ExcelWriter(fname) as excel:
		for strategy in overall:
			df = pd.DataFrame.from_dict(overall[strategy])
			df.to_excel(excel, sheet_name=strategy)


def acc_vs_time_vs_iters(probs, configs, costs, n_servers):
	"""
	Compute scalability data for accuracy vs time vs iters
	"""
	overall = {'acc': [], 'runtime': [], 'iters': [], 'predictive': []}
	for per in range(5, 101, 5):
		acc = per / 100
		results = simulater(probs, configs, costs, n_servers, 'wrr', acc)

		preds = []
		runtimes = []
		for k, result in results.items():
			preds += [result['predictive']]
			runtimes += [result['runtime']]

		pred = sum(preds) / len(preds)
		runtime = sum(runtimes)

		overall['acc'] += [acc]
		overall['runtime'] += [runtime]
		overall['iters'] += [len(results)]
		overall['predictive'] += [pred]

	fname = join(out_dir, 'acc_vs_time_vs_iters.xlsx')
	df = pd.DataFrame.from_dict(overall)
	df.to_excel(fname)


if __name__ == '__main__':
	args = get_args()
	out_dir = args.out_dir
	success_rate = args.success_rate / 100

	hashes, probs = get_data(args.p_csv)
	print('# of samples: ', len(hashes))

	# print('Scheduling a single run...')
	# results = simulater(probs, config_names, config_costs, args.n_servers, args.strategy, success_rate)
	# preds = []
	# runtimes = []
	# for k, result in results.items():
	#     preds += [result['predictive']]
	#     runtimes += [result['runtime']]
	#
	# pred = sum(preds) / len(preds)
	# runtime = sum(runtimes)
	# print('** Average predictive value: %.2f' % pred)
	# print('** Total runtime: %d' % runtime)

	print('Computing server vs time...')
	server_vs_time(probs, config_names, config_costs, args.n_servers, success_rate)

	# print('Computing accuracy vs time vs iters...')
	# acc_vs_time_vs_iters(probs, config_names, config_costs, args.n_servers)
