"""
Scheduling samples to configs based on output of the ML multilabel
classifier. 

Key considerations: 
- load balancing

** If it fails, update probs / labels and rerun scheduling algo
to find where sample goes next.
"""

import argparse
import pandas as pd
import numpy as np
from itertools import compress

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

config_costs = {  # startup, runtime, resources
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


def get_args():
    parser = argparse.ArgumentParser(description='Scheduling args')

    parser.add_argument('--p_csv', '-p', type=str, required=True, help='path to CSV containing model output')
    parser.add_argument('--n_servers', '-ns', type=int, default=13, help='number of servers')

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


def balancer(loads, samples):
    """
    Balance loads while maximizing sample execution probability
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


def scheduler(probs, configs, costs, n_servers):
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
    schedule = {server: [] for server in range(n_servers)}

    ### PART 1: ALLOCATION
    print('Allocating...')

    # build sample info sheet
    samples = {}
    labels = threshold(probs)
    for ind, (prob, label) in enumerate(zip(probs, labels)):
        # filter each samples predicted configs and probs of success
        conf_names = list(compress(configs, label))
        conf_probs = list(compress(prob, label))

        # sort based on prob (descending) and save to dict
        sorted_info = sorted(zip(conf_names, conf_probs), key=lambda a: a[1], reverse=True)
        samples[ind] = {'configs': [n for n, _ in sorted_info], 'probs': [p for _, p in sorted_info]}

    # # sort columns based on prob (ascending) then flip for descending
    # sorted_inds = np.flipud(np.argsort(probs, axis=0))

    # round robin allocation with load balancing
    C = len(samples) // len(configs)  # soft limit per config
    print('Soft limit per config: ', C)

    negatives = []
    loads = {config: [] for config in configs}
    for ind, sample in samples.items():
        # s_probs = sample['probs']
        s_confs = sample['configs']

        # record negative prediction sample
        if not s_confs:
            negatives += [ind]

        # weighted allocation
        else:
            allocated = False
            for conf in s_confs:
                if len(loads[conf]) < C:
                    loads[conf] += [ind]
                    allocated = True

            # if sample couldn't be allocated, just allocate to likeliest config
            if not allocated:
                mx_conf = s_confs[0]
                loads[mx_conf] = [ind]

    print('# of negatives: ', len(negatives))
    for config, load in loads.items():
        print('# of samples allocated to %s: ' % config, len(load))

    # load balancing
    balancer(loads, samples)

    ### PART 2: SCHEDULING
    print('Scheduling to servers...')

    return schedule, negatives


if __name__ == '__main__':
    args = get_args()

    print('Getting data...')
    hashes, probs = get_data(args.p_csv)

    print('Starting scheduler...')
    schedule, negatives = scheduler(probs, config_names, config_costs, args.n_servers)
    neg_samples = hashes[negatives]

    # compute average success prob total and on each server
    # compute total runtime (aka longest running server)
