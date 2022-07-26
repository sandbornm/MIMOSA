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
    # round robin allocation with load balancing
    C = len(samples) // len(configs)  # soft limit per config
    print('Soft limit per config: ', C)

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
            for conf, prob in sorted_info:
                if not allocated and len(loads[conf]) < C:
                    loads[conf]['inds'] += [ind]
                    loads[conf]['probs'] += [prob]
                    allocated = True

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

        # record negative prediction sample
        if sum(s_labels) == 0:
            negatives += [ind]

        # assign randomly regardless
        j = choice(n_configs)
        rand_conf = configs[j]

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

    negatives = []
    loads = {config: {'inds': [], 'probs': []} for config in configs}
    for ind, sample in samples.items():
        s_probs = sample['probs']
        s_labels = sample['labels']

        # record negative prediction sample
        if sum(s_labels) == 0:
            negatives += [ind]

        # assign to supposed best conf
        loads[best_conf]['inds'] += [best_ind]
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


def balancer(loads, samples):
    """
    **UNFINISHED**
    Balance loads while maximizing probability and minimizing runtime
    """
    print('Balancing...')

    # find min and max loads
    lens = {}
    mx = None
    mx_conf = ""
    mn = None
    mn_conf = ""
    for config, load in loads.items():
        n_samples = len(load)
        lens[config] = n_samples
        if not mx or n_samples > mx:
            mx = n_samples
            mx_conf = config
        if not mn or n_samples < mn:
            mn = n_samples
            mn_conf = config

    # rebalance excess samples to other configs
    diff = mx - mn
    mx_load = loads[mx_conf]
    base, excess = mx_load[:len(mx_load) - diff], mx_load[len(mx_load) - diff:]

    # recursive rebalancing??

    return


def scheduler(probs, configs, costs, n_servers, strategy='wrr'):
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
    labels = threshold(probs)
    samples = {ind: {'probs': prob, 'labels': label} for ind, (prob, label) in enumerate(zip(probs, labels))}

    # sort columns based on prob (ascending) then flip for descending
    # sorted_inds = np.flipud(np.argsort(probs, axis=0))

    ### PART 1: ALLOCATION
    print('Allocating...')

    loads, negatives = allocator(samples, configs, strategy)
    print('# of negatives: ', len(negatives))

    # load balancing
    # balancer(loads, samples)

    ### PART 2: SCHEDULING
    print('Scheduling to servers...')

    # compute costs for each load as in runtime and resources as a product each sample in load
    # treating loads as static units
    # assumption: sample domain randomly selected --> extremely unlikely to predict only one config

    for config, load in loads.items():
        s_inds = load['inds']
        # s_probs = load['probs']
        startup, runtime, resources = costs[config]

        n_samples = len(s_inds)
        print('# of samples allocated to %s: ' % config, n_samples)

        if n_samples == 0:
            continue

        load_runtime = startup + n_samples * runtime
        print('>>> with total runtime: %d' % load_runtime)
        print('>>> with resources: %d' % resources)

        # greedy assignment to server based on current run time
        server = min(schedule, key=lambda server: schedule[server]['runtime'])

        print('>>> assigned to server %d' % server)

        schedule[server]['configs'] += [config]
        schedule[server]['loads'] += [load]
        schedule[server]['costs'] += [(load_runtime, resources)]
        schedule[server]['runtime'] += load_runtime

    return schedule, negatives


def simulate(probs, configs, costs, n_servers, strategy, success=1.0):
    """
    Simulate a run of scheduling based on servers and success rate
    """
    print('** Strategy is %s' % strategy)

    rem = list(range(len(probs)))
    results = {}
    k = 0
    data = probs.copy()
    while rem:
        k += 1

        print('\n\n** Iteration: ', k)
        schedule, negatives = scheduler(data[rem, :], configs, costs, n_servers, strategy)
        # print('Schedule: \n', schedule)
        if len(negatives) == len(rem):  # empty schedule bc all failed so done
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
        results[k] = {'predictive': mean,
                   'runtime': total_runtime}

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
    strategies = ['wrr', 'random', 'koth', 'mimosa']
    overall = {strategy: {'n_servers': [], 'predictive': [], 'runtime': [], 'iters': []} for strategy in strategies}
    for strategy in strategies:
        for n_servers in range(1, max_servers+1):
            if strategy == 'mimosa':
                results = simulate(probs, configs, costs, n_servers, 'wrr', success)
            else:
                results = simulate(probs, configs, costs, n_servers, strategy)

            preds = []
            runtimes = []
            for k, result in results.items():
                preds += [result['predictive']]
                runtimes += [result['runtime']]

            pred = sum(preds) / len(preds)
            runtime = sum(runtimes)

            overall[strategy]['n_servers'] += [n_servers]
            overall[strategy]['runtime'] += [runtime]
            overall[strategy]['iters'] += [len(results)]
            overall[strategy]['predictive'] += [pred]

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
        results = simulate(probs, configs, costs, n_servers, 'wrr', acc)

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
    # results = simulate(probs, config_names, config_costs, args.n_servers, args.strategy, success_rate)
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

    print('Computing accuracy vs time vs iters...')
    acc_vs_time_vs_iters(probs, config_names, config_costs, args.n_servers)
