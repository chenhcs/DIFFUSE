from keras.preprocessing import sequence
import numpy as np
from sys import argv


def domain_id_map():
    fr = open('../data/raw_data/domain_data/domain_id_mapping.txt')
    entry = fr.readline().split('\n')[0]
    domain_dic = {}
    while entry != '':
        value, key = entry.split('\t')
        domain_dic[key] = int(value)
        entry = fr.readline().split('\n')[0]
    fr.close()

    domain_dic['0'] = 0
    return domain_dic


def iso_domain_map(domain_dic):
    fr = open('../data/raw_data/domain_data/human_isoform_dm.txt')
    entry = fr.readline().split('\n')[0]
    isofDmDic = {}
    while entry != '':
        isof, domains = entry.split('\t')[1:]
        domains += '0'
        tmpD = [domain_dic[key] for key in domains.split(' ')]
        if len(tmpD) > 1:
            tmpD.pop()
        print tmpD
        isofDmDic[isof] = tmpD
        entry = fr.readline().split('\n')[0]
    fr.close()

    fr = open('../data/raw_data/domain_data/swissprot_isoform_dm.txt')
    entry = fr.readline().split('\n')[0]
    while entry != '':
        gene, prot, domains = entry.split('\t')
        domains += '0'
        tmpD = [domain_dic[key] for key in domains.split(' ')]
        if len(tmpD) > 1:
            tmpD.pop()
        print gene, tmpD
        isofDmDic[gene] = tmpD
        entry = fr.readline().split('\n')[0]
    fr.close()
    return isofDmDic


def make_dataset(isofDmDic):
    HUMANX_train_domain = []
    HUMANX_test_domain = []
    HUMANtrain_iso_id_ho_s = np.load('../data/id_lists/train_isoform_list.npy')
    HUMANtest_iso_id_ho_s = np.load('../data/id_lists/test_isoform_list.npy')
    max_length = 0
    for isoid in HUMANtrain_iso_id_ho_s:
        #isoid = isoid.split(',')[0]
        if isoid in isofDmDic.keys():
            HUMANX_train_domain.append(isofDmDic[isoid])
            if len(isofDmDic[isoid]) > max_length:
                max_length = len(isofDmDic[isoid])
        else:
            print isoid
            HUMANX_train_domain.append([0])

    for isoid in HUMANtest_iso_id_ho_s:
        #isoid = isoid.split(',')[0]
        if isoid in isofDmDic.keys():
            HUMANX_test_domain.append(isofDmDic[isoid])
            if len(isofDmDic[isoid]) > max_length:
                max_length = len(isofDmDic[isoid])
        else:
            print isoid
            HUMANX_test_domain.append([0])

    OTHERtrain_iso_id_ho_s = np.load('../data/id_lists/train_swissprot_list.npy')
    OTHERX_train_domain = []
    cnt = 1
    for isoid in OTHERtrain_iso_id_ho_s:
        #isoid = isoid.split(',')[0]
        print cnt
        cnt += 1
        if isoid in isofDmDic.keys():
            OTHERX_train_domain.append(isofDmDic[isoid])
            if len(isofDmDic[isoid]) > max_length:
                max_length = len(isofDmDic[isoid])
        else:
            OTHERX_train_domain.append([0])

    print max_length
    HUMANX_train_np = np.array(sequence.pad_sequences(HUMANX_train_domain, max_length))
    HUMANX_test_np = np.array(sequence.pad_sequences(HUMANX_test_domain, max_length))
    OTHERX_train_np = np.array(sequence.pad_sequences(OTHERX_train_domain, max_length))

    return HUMANX_train_np, HUMANX_test_np, OTHERX_train_np


domain_dic = domain_id_map()
isofDmDic = iso_domain_map(domain_dic)
HUMANX_train_np, HUMANX_test_np, OTHERX_train_np = make_dataset(isofDmDic)
print HUMANX_train_np.shape
print HUMANX_test_np.shape
print OTHERX_train_np.shape
np.save('../data/domains/human_domain_train.npy', HUMANX_train_np)
np.save('../data/domains/human_domain_test.npy', HUMANX_test_np)
np.save('../data/domains/swissprot_domain_train.npy', OTHERX_train_np)
