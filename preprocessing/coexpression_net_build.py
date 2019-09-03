import numpy as np
from numpy import genfromtxt

def coexpression_net(study_id, power):
    exp_mat = genfromtxt('../data/raw_data/expression_data/expression_mat_' + study_id + '.csv', delimiter=',')
    exp_mat = exp_mat[1:, 1:]

    # Calculate co-exp net
    cor_net = np.corrcoef(exp_mat)

    # Set nan to be zero
    nan_where = np.isnan(cor_net)
    cor_net[nan_where] = 0

    # Diagnal to be zero
    for i in range(cor_net.shape[0]):
        cor_net[i, i] = 0

    # Apply soft threshold
    cor_net = np.fabs(cor_net)
    cor_net = pow(cor_net, power)

    np.save('../data/raw_data/expression_data/coexp_net_' + study_id + '.npy', cor_net)

    return cor_net.shape[0]


def unify_all_nets(number_isoform_nodes, sra_study_list, cor_net_unify_file):
    cor_net_unify = np.zeros([number_isoform_nodes, number_isoform_nodes])

    for study_id in sra_study_list:
        cor_net = np.load('../data/raw_data/expression_data/coexp_net_' + study_id + '.npy')

        # Normalize each row
        mean_rows = np.mean(cor_net, axis = 1)
        for i in range(cor_net.shape[0]):
            if mean_rows[i] == 0:
                continue
            cor_net[i, :] = cor_net[i, :] / mean_rows[i]

        print np.mean(cor_net, axis = 1)
        print cor_net
        cor_net_unify += cor_net

        print cor_net_unify

    # Normalize each row of the unified net
    mean_rows_unify = np.mean(cor_net_unify, axis = 1)
    for i in range(cor_net_unify.shape[0]):
        if mean_rows_unify[i] == 0:
            continue
        cor_net_unify[i, :] = cor_net_unify[i, :] / mean_rows_unify[i]
    print np.mean(cor_net_unify, axis = 1)

    cor_net_unify = pow(cor_net_unify, 8)

    # Normalize each row of the unified net
    mean_rows_unify = np.mean(cor_net_unify, axis = 1)
    for i in range(cor_net_unify.shape[0]):
        if mean_rows_unify[i] == 0:
            continue
        cor_net_unify[i, :] = cor_net_unify[i, :] / mean_rows_unify[i]
    print np.mean(cor_net_unify, axis = 1)

    print cor_net_unify

    np.save(cor_net_unify_file, cor_net_unify)

def read_power(power_threshold_file):
    fr = open(power_threshold_file, 'r')

    sra_power_map = {}
    while True:
        line = fr.readline()
        if not line:
            break
        sra_id, power = line.split('\n')[0].split('\t')
        sra_power_map[sra_id] = int(power)

    fr.close()
    return sra_power_map

if __name__=='__main__':
    power_threshold_file = '../data/raw_data/expression_data/power_thresholds.txt'
    sra_power_map = read_power(power_threshold_file)
    for key in sra_power_map:
        print key, sra_power_map[key]

    for study_id in sra_power_map.keys():
        number_isoform_nodes = coexpression_net(study_id, sra_power_map[study_id])

    K_number_isoform_nodes = 39375
    cor_net_unify_file = '../data/co-expression_net/coexp_net_unified.npy'
    unify_all_nets(K_number_isoform_nodes, sra_power_map.keys(), cor_net_unify_file)
