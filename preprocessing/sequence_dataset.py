from keras.preprocessing import sequence
import numpy as np
from sys import argv

train_iso_ids = np.load('../data/id_lists/train_isoform_list.npy')
test_iso_ids = np.load('../data/id_lists/test_isoform_list.npy')
train_prot_id = np.load('../data/id_lists/train_swissprot_list.npy')

aa_num_dict = {}

aa_num_dict['F'] = 1
aa_num_dict['L'] = 2
aa_num_dict['I'] = 3
aa_num_dict['M'] = 4
aa_num_dict['V'] = 5
aa_num_dict['S'] = 6
aa_num_dict['P'] = 7
aa_num_dict['T'] = 8
aa_num_dict['A'] = 9
aa_num_dict['Y'] = 10
aa_num_dict['H'] = 11
aa_num_dict['Q'] = 12
aa_num_dict['N'] = 13
aa_num_dict['K'] = 14
aa_num_dict['D'] = 15
aa_num_dict['E'] = 16
aa_num_dict['C'] = 17
aa_num_dict['W'] = 18
aa_num_dict['R'] = 19
aa_num_dict['G'] = 20

print('Converting trigrams to numbers...')
X_other_train = []
protein_count = 0
truncation_length = 6000
max_len = 0

fr = open('../data/raw_data/sequence_data/swissprot_protein_sequences.txt', 'r')
while True:
    line = fr.readline().split('\n')[0]
    if not line:
        break
    if '>' in line:
        prot_id = line.split('|')[1]
        prot_seq = fr.readline().split('\n')[0]
        if prot_id in train_prot_id:
            numseq = []
            for j in range(len(prot_seq) - 2):
                ngram = (aa_num_dict[prot_seq[j]] - 1) * 400 + (aa_num_dict[prot_seq[j + 1]] - 1) * 20 + (aa_num_dict[prot_seq[j + 2]])
                numseq.append(ngram)
            if len(numseq) > max_len:
                max_len = len(numseq)
            if len(numseq) > truncation_length:
                numseq = numseq[:truncation_length]
            X_other_train.append(numseq)
            protein_count += 1
            print prot_id
fr.close()

X_train = []
X_test = []
iso_count = 0
fr = open('../data/raw_data/sequence_data/isoform_cds_sequences.txt', 'r')
while True:
    line = fr.readline().split('\n')[0]
    if not line:
        break
    if '>' in line:
        iso_id = line.split('|')[1]
        iso_seq = fr.readline().split('\n')[0]
        numseq = []
        for j in range(len(iso_seq) - 2):
            ngram = (aa_num_dict[iso_seq[j]] - 1) * 400 + (aa_num_dict[iso_seq[j + 1]] - 1) * 20 + (aa_num_dict[iso_seq[j + 2]])
            numseq.append(ngram)
        if len(numseq) > max_len:
            max_len = len(numseq)
        if len(numseq) > truncation_length:
            numseq = numseq[:truncation_length]
        if iso_id in train_iso_ids:
            X_train.append(numseq)
            iso_count += 1
            print iso_id
        elif iso_id in test_iso_ids:
            X_test.append(numseq)
            iso_count += 1
            print iso_id
fr.close()

print('Complete convertion.')
max_len = min(max_len, truncation_length)
print(max_len)
X_train_np = np.array(sequence.pad_sequences(X_train, max_len))
X_test_np = np.array(sequence.pad_sequences(X_test, max_len))
X_other_np = np.array(sequence.pad_sequences(X_other_train, max_len))
print X_train_np.shape
print X_test_np.shape
print X_other_np.shape
np.save('../data/sequences/human_sequence_train.npy', X_train_np)
np.save('../data/sequences/human_sequence_test.npy', X_test_np)
np.save('../data/sequences/swissprot_sequence_train.npy', X_other_np)
