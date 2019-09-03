import numpy as np

def generate_label(X_test_geneid, positive_Gene):
    y_test = np.array([])

    last_gID = ''
    for gID in X_test_geneid:
        if gID != last_gID:
            if gID in positive_Gene:
                y_test = np.hstack((y_test, np.ones(1)))
            else:
                y_test = np.hstack((y_test, np.zeros(1)))
            last_gID = gID
        else:
            y_test = np.hstack((y_test, y_test[-1]))

    return y_test

def make_batch(seqX):
    gpl_dic = {}
    nonspace = np.sign(seqX)
    aalen = np.sum(nonspace, 1)
    idx = np.argsort(aalen)
    len_srt = aalen[idx]
    stidx = 0
    gp_n = 0
    maxlen = 1000
    add_length = maxlen * 2
    for i in range(seqX.shape[0]):
        if len_srt[i] > maxlen:
            gpl_dic[gp_n] = (stidx, i - 1, maxlen)
            gp_n += 1
            stidx = i
            maxlen += add_length
            add_length *= 2
    gpl_dic[gp_n] = (stidx, seqX.shape[0] - 1, maxlen)
    return idx, gpl_dic
