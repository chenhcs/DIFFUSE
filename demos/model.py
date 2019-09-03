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
from utils import generate_label, make_batch
import time
from sys import argv
script, GOterm = argv

def run_crf(score_map, co_exp_net, testing_size, theta):
    positive_unary_energy = 1 - score_map
    crf = CRF(testing_size, positive_unary_energy, co_exp_net, theta)
    pos_prob_crf = crf.inference(10)

    return pos_prob_crf

#-------------------------------------------------------------------------------
#Loading sequence data
def load_sequence_data():
    X_test_seq = np.load('../data/sequences/human_sequence_test_demo.npy')
    X_test_dm = np.load('../data/domains/human_domain_test_demo.npy')
    X_test_geneid = np.load('../data/id_lists/gene_list_test_demo.npy')
    X_test_isoid = np.load('../data/id_lists/isoform_list_test_demo.npy')

    return X_test_seq, X_test_dm, X_test_geneid, X_test_isoid

#-------------------------------------------------------------------------------
#Loading positive and negtive set
def pos_gene_set(StudiedGOterm):
    positive_set = []

    fr = open('../data/annotations/human_annotations.txt', 'r')
    while True:
        line = fr.readline()
        if not line:
            break
        line = line.split('\n')[0]
        gene = line.split('\t')[0]
        GO = line.split('\t')[1:]
        if StudiedGOterm in GO:
            positive_set.append(gene)
    fr.close()

    return positive_set

#-------------------------------------------------------------------------------
###### main function ######
StudiedGOterm = GOterm

print('Testing model for ' + StudiedGOterm)

positive_Gene = pos_gene_set(StudiedGOterm)

co_exp_net = np.load('../data/co-expression_net/coexp_net_unified_demo.npy')
X_test_seq, X_test_dm, X_test_geneid, X_test_isoid = load_sequence_data()
K_testing_size = X_test_seq.shape[0]
seq_dim = X_test_seq.shape[1]
dm_dim = X_test_dm.shape[1]

print('Generating initial label...')
y_test = generate_label(X_test_geneid, positive_Gene)

## Model architecture
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

adam = optimizers.Adam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay=0.01)
model.compile(loss= losses.binary_crossentropy, optimizer= adam, metrics=['accuracy'])

## Load model
model.load_weights('../saved_models/'+ StudiedGOterm + '_DNN.h5')
theta = np.load('../saved_models/'+ StudiedGOterm +'_CRF_weights.npy')

## Testing
tup_idx, tup_gp = make_batch(X_test_seq)
y_pred = np.array([])
for key in tup_gp.keys():
    sel = tup_gp[key]
    st = sel[0]
    ed = sel[1]
    le = sel[2]
    X_test_seq_batch = X_test_seq[tup_idx[st: ed + 1]]
    X_test_dm_batch = X_test_dm[tup_idx[st: ed + 1]]
    for i in range(X_test_seq_batch.shape[0] / 1000 + 1):
        y_pred_l = model.predict_on_batch([X_test_seq_batch[1000 * i : 1000 * (i + 1), seq_dim - le: seq_dim], X_test_dm_batch[1000 * i : 1000 * (i + 1)]])
        y_pred = np.hstack((y_pred, np.transpose(y_pred_l)[0]))
ori_indx = np.argsort(tup_idx)
initial_score = y_pred[ori_indx]

print("Run mean field approximation for CRF.")
pos_prob_crf = run_crf(initial_score, co_exp_net, K_testing_size, theta)

y_test_gene = []
y_pred_gene = []
y_pred = pos_prob_crf
for gid in set(list(X_test_geneid)):
    idx = np.where(X_test_geneid == gid)
    y_test_gene.append(np.max(y_test[idx]))
    y_pred_gene.append(np.max(y_pred[idx]))
auc_roc = roc_auc_score(y_test_gene, y_pred_gene)
print('\nPerformance evaluation on test data:\nAUC:' + str(auc_roc))

precision, recall, thresholds = precision_recall_curve(y_test_gene, y_pred_gene)
auc_prc = auc(recall, precision)
y_test_gene = np.array(y_test_gene)
ratio = len(np.where(y_test_gene == 1)[0]) * 1.0 / len(y_test_gene)
print('AUPRC:' + str(auc_prc) + '(baseline:' + str(ratio) + ')')

# Print predictions
print("\nThe predictions for test data are saved in the results folder.")
fw = open('../results/' + StudiedGOterm + '_prediction_scores.txt', 'w')
for i in range(K_testing_size):
    fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t')
    fw.write(str(pos_prob_crf[i]) + '\n')
fw.close()
