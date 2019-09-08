import numpy as np
import random

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, SpatialDropout1D
from keras.layers import Activation, Flatten, Input, Masking
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import regularizers
from keras import losses
from keras import optimizers
from keras.models import Model
from crf import CRF
from layer.PyramidPooling import PyramidPooling
from keras.utils import multi_gpu_model
from utils import generate_label, upsample, make_batch
import time
from sys import argv
script, go_index = argv


def run_crf(score_map, bag_label, bag_index, co_exp_net, training_size, testing_size, theta, sigma = 10):
    bag_label = bag_label[0: training_size]
    bag_index = bag_index[0: training_size]
    positive_unary_energy = 1 - score_map

    crf = CRF(training_size, testing_size, positive_unary_energy, co_exp_net, theta, bag_label, bag_index)

    label_update, pos_prob_crf, unary_potential, pairwise_potential = crf.inference(10)
    theta_prime = crf.parameter_learning(label_update, theta, sigma)

    return label_update, theta_prime, pos_prob_crf, unary_potential, pairwise_potential


def find_GO(num):
    fr = open('../data/go_terms/go.txt')
    for _ in range(num):
        go_term, _ = fr.readline().split('\n')[0].split('\t')
    fr.close()

    return go_term


#Load sequence data
def load_sequence_data():
    X_train_seq = np.load('../data/sequences/human_sequence_train.npy')
    X_train_dm = np.load('../data/domains/human_domain_train.npy')
    X_test_seq = np.load('../data/sequences/human_sequence_test.npy')
    X_test_dm = np.load('../data/domains/human_domain_test.npy')
    X_train_geneid = np.load('../data/id_lists/train_gene_list.npy')
    X_train_geneid = list(X_train_geneid)
    X_test_geneid = np.load('../data/id_lists/test_gene_list.npy')
    X_test_geneid = list(X_test_geneid)
    X_test_isoid = np.load('../data/id_lists/test_isoform_list.npy')
    X_test_isoid = list(X_test_isoid)

    X_train_other_seq = np.load('../data/sequences/swissprot_sequence_train.npy')
    X_train_other_dm = np.load('../data/domains/swissprot_domain_train.npy')
    X_train_geneid_other = np.load('../data/id_lists/train_swissprot_list.npy')
    X_train_geneid_other = list(X_train_geneid_other)

    return X_train_seq, X_train_dm, X_test_seq, X_test_dm, X_train_geneid, X_test_geneid, X_test_isoid, X_train_other_seq, X_train_other_dm, X_train_geneid_other


#Load positive and negtive set
def pos_gene_set(selected_go):
    positive_set = []

    fr = open('../data/annotations/human_annotations.txt', 'r')
    while True:
        line = fr.readline()
        if not line:
            break
        line = line.split('\n')[0]
        gene = line.split('\t')[0]
        GO = line.split('\t')[1:]
        if selected_go in GO:
            positive_set.append(gene)
    fr.close()

    fr = open('../data/annotations/swissprot_annotations.txt')
    while True:
        line = fr.readline()
        if not line:
            break
        line = line.split('\n')[0]
        prot = line.split('\t')[0]
        GO = line.split('\t')[1:]
        if selected_go in GO:
            positive_set.append(prot)
    fr.close()

    return positive_set

###### main function ######
selected_go = find_GO(int(go_index))
print('Training model for ' + selected_go)

positive_Gene = pos_gene_set(selected_go)

co_exp_net = np.load('../data/co-expression_net/coexp_net_unified.npy')
X_train_seq, X_train_dm, X_test_seq, X_test_dm, X_train_geneid, X_test_geneid, X_test_isoid, X_train_other_seq, X_train_other_dm, X_train_geneid_other = load_sequence_data()
K_training_size = X_train_seq.shape[0]
K_testing_size = X_test_seq.shape[0]
X_all_seq = np.vstack([X_train_seq, X_test_seq])
X_all_dm = np.vstack([X_train_dm, X_test_dm])
seq_dim = X_train_seq.shape[1]
dm_dim = X_train_dm.shape[1]

print('Generating initial label...')
y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, X_train_seq, X_train_dm= generate_label(X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm, X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

#Model architecture
seq_input = Input(shape=(None, ), dtype='int32', name='seq_input')
x1 = Embedding(input_dim = 8001, output_dim = 32)(seq_input)
x1 = Convolution1D(filters = 64, kernel_size = 32, strides = 1,  padding = 'valid', activation = 'relu')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=regularizers.l2(0.15))(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(16, kernel_regularizer=regularizers.l2(0.15))(x1)
seq_output = Activation('relu')(x1)

domain_input = Input(shape=(dm_dim, ), dtype='int32', name='domain_input')
x2 = Embedding(input_dim = 16155, output_dim = 32, input_length = dm_dim, mask_zero = True)(domain_input)
domain_output = LSTM(16)(x2)
x = keras.layers.concatenate([seq_output, domain_output])

x = Dense(16, kernel_regularizer=regularizers.l2(0.15))(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, kernel_regularizer=regularizers.l2(0.15))(x)
output = Activation('sigmoid')(x)

model = Model(inputs = [seq_input, domain_input], output = output)

model.summary()

adam = optimizers.Adam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay=0.1)
model.compile(loss= losses.binary_crossentropy, optimizer= adam, metrics=['accuracy'])


#Training
nb_epoch = 5

unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)
X_test_geneid = np.array(X_test_geneid)

theta = np.array([1.0, 1.0])

for epoch in range(nb_epoch):
    print('epoch:', epoch)
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    for e in range(3):
        for key in tup_gp.keys():
            sel = tup_gp[key]
            st = sel[0]
            ed = sel[1]
            le = sel[2]
            X_train_seq_batch = X_train_seq_upsmp[tup_idx[st: ed + 1]]
            X_train_dm_batch = X_train_dm_upsmp[tup_idx[st: ed + 1]]
            y_train_batch = y_train_upsmp[tup_idx[st: ed + 1]]
            positive_index = np.where(y_train_batch == 1)
            negtive_index = np.where(y_train_batch == 0)
            index = np.hstack((positive_index[0], negtive_index[0]))
            if index.shape[0] == 0:
                continue
            np.random.shuffle(index)
            if key == 0:
                model.fit([X_train_seq_batch[index, seq_dim - np.min((le, seq_dim)): seq_dim], X_train_dm_batch[index]], y_train_batch[index], batch_size=1024, epochs=1, verbose=1)
            elif key == 1:
                model.fit([X_train_seq_batch[index, seq_dim - np.min((le, seq_dim)): seq_dim], X_train_dm_batch[index]], y_train_batch[index], batch_size=512, epochs=1, verbose=1)
            else:
                model.fit([X_train_seq_batch[index, seq_dim - np.min((le, seq_dim)): seq_dim], X_train_dm_batch[index]], y_train_batch[index], batch_size=256, epochs=1, verbose=1)

    tup_idx, tup_gp = make_batch(X_all_seq)
    initial_score_all = np.array([])
    for key in tup_gp.keys():
        sel = tup_gp[key]
        st = sel[0]
        ed = sel[1]
        le = sel[2]
        X_all_seq_batch = X_all_seq[tup_idx[st: ed + 1]]
        X_all_dm_batch = X_all_dm[tup_idx[st: ed + 1]]
        for i in range(X_all_seq_batch.shape[0] / 1000 + 1):
            initial_score_all_ = model.predict_on_batch([X_all_seq_batch[1000 * i: 1000 * (i + 1), seq_dim - le: seq_dim], X_all_dm_batch[1000 * i : 1000 * (i + 1)]])
            initial_score_all = np.hstack((initial_score_all, np.transpose(initial_score_all_)[0]))
    ori_indx = np.argsort(tup_idx)
    initial_score_all = initial_score_all[ori_indx]

    y_train[0: K_training_size], theta, pos_prob_crf, unary_potential, pairwise_potential = run_crf(initial_score_all, y_all, crf_bag_index, co_exp_net, K_training_size, K_testing_size, theta, sigma = 0.1)

    if epoch < nb_epoch - 1:
        X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# Testing
y_test_gene = []
y_pred_gene = []
y_test0 = y_test
y_pred0 = pos_prob_crf[K_training_size: K_training_size + K_testing_size]
X_test_geneid0 = X_test_geneid
for gid in set(list(X_test_geneid0)):
    idx = np.where(X_test_geneid0 == gid)
    y_test_gene.append(np.max(y_test0[idx]))
    y_pred_gene.append(np.max(y_pred0[idx]))
auc_roc_crf = roc_auc_score(y_test_gene, y_pred_gene)
print('AUC:' + str(auc_roc_crf))

# Save model
model.save('../saved_models/' + selected_go + '_DNN.h5')
np.save('../saved_models/' + selected_go + '_CRF_weights.npy', theta)

# Write predictions
fw = open('../results/' + selected_go + '_prediction_scores.txt', 'w')
for i in range(K_testing_size):
    fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t')
    fw.write(str(pos_prob_crf[K_training_size + i]) + '\n')
fw.close()
