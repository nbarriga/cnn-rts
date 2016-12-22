#!/usr/bin/env python
import sys
sys.path.append("~/git-working/caffe/python/")

import numpy as np
import lmdb
import caffe
import time
import math
import shutil

start_time = time.time()
np.random.seed(24)


def read_games(channels, dimension, nrGames, path, process_labels):
    data = np.zeros((nrGames, channels, dimension, dimension), dtype=np.uint8)
    labels = np.zeros(nrGames, dtype=np.uint8)

    for i in range(nrGames):
        if i % 5000 == 0:
            print i

        array = np.fromfile(path+str(i), dtype=np.uint8, count=-1, sep=" ")
        w = array[0]
        h = array[1]
        c = array[2]

        assert w == h and w == dimension and c == channels

        labels[i] = array[-1]

        if process_labels:
            t = int(labels[i])/25
            if t == 2:
                labels[i] = 2  # draw
            elif t < 2:
                labels[i] = 0  # defeat
            else:
                labels[i] = 1  # victory

        array = np.delete(array, array.size-1)
        array = np.delete(array, [0, 1, 2])
        # print array.size
        data[i] = np.reshape(array, [c, w, h])
    return data, labels


# save_path = '/media/root/EXTERNAL_2GB/'
def create_lmdb(data, labels, save_path, name, extra, mean):
    buffer_size = 100
    DB_KEY_FORMAT = "{:0>10d}"

    map_size = data[0].astype(float).nbytes * 2 * len(labels) + 100
    lmdb_name = save_path + name + extra

    c = data[0].shape[0]
    h = data[0].shape[1]
    w = data[0].shape[2]

    try:
        shutil.rmtree(lmdb_name)
    except:
        print "error deleting"

    for idx in range(int(math.ceil(len(labels)/float(buffer_size)))):
        if idx % 10 == 0:
            print idx*buffer_size
        in_db = lmdb.open(lmdb_name, map_size=map_size)
        with in_db.begin(write=True) as in_txn:

            for current_idx, array in enumerate(data[(buffer_size*idx):(buffer_size*(idx+1))]):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = c
                datum.height = h
                datum.width = w
                # TODO: check current idx is correct
                datum.label = int(labels[buffer_size*idx + current_idx])
                # datum.data = ((array.astype(float) - mean)*128.0).tobytes()
                datum.data = array.tobytes()

                str_id = DB_KEY_FORMAT.format(buffer_size*idx + current_idx)
                in_txn.put(str_id, datum.SerializeToString())
        in_db.close()
    return


def shuffle(data, labels):
    assert len(data) == len(labels)
    p = np.random.permutation(len(labels))
    return data[p], labels[p]

def shuffle_s(data, labels, samples):
    assert len(data) == len(labels)
    realLen = int(len(labels) / samples)
    p = np.random.permutation(realLen)
    pp = np.zeros(realLen*samples).astype(int)
    for i in range(realLen):
        pp[i*samples:(i+1)*samples] = np.arange(samples) + samples*p[i]
    # print pp
    return data[pp], labels[pp]


def augment(data, labels):

    aug_data = np.zeros((len(labels)*8, data[0].shape[0], data[0].shape[1], data[0].shape[2]), dtype=np.uint8)
    aug_labels = np.zeros((len(labels)*8), dtype=np.uint8)

    for idx, array in enumerate(data):
        aug_labels[idx*8:(idx+1)*8] = labels[idx]
        for i in xrange(array.shape[0]):
            aug_data[idx*8+0][i] = array[i]
            aug_data[idx*8+1][i] = np.rot90(array[i])
            aug_data[idx*8+2][i] = np.rot90(array[i], 2)
            aug_data[idx*8+3][i] = np.rot90(array[i], 3)
            aug_data[idx*8+4][i] = np.flipud(aug_data[idx*8+0][i])
            aug_data[idx*8+5][i] = np.fliplr(aug_data[idx*8+1][i])
            aug_data[idx*8+6][i] = np.flipud(aug_data[idx*8+2][i])
            aug_data[idx*8+7][i] = np.fliplr(aug_data[idx*8+3][i])
    return aug_data, aug_labels

totalGames = 2000
samples = 3
meanD = 0

d, l = read_games(25, 24, totalGames, '../microrts/24x24extracted/game', 0)

ds, ls = shuffle_s(d, l, samples)

test_start = int(totalGames * 0.9)

dtst = ds[test_start:totalGames]
ltst = ls[test_start:totalGames]

print("test: %s seconds ---" % (time.time() - start_time))
create_lmdb(dtst, ltst, './data/', 'test', '24x24', meanD)

d = ds[0:test_start]
l = ls[0:test_start]

print("augmenting: %s seconds ---" % (time.time() - start_time))
dsa, lsa = augment(d, l)
print("shuffling: %s seconds ---" % (time.time() - start_time))
d2, l2 = shuffle(dsa, lsa)

# meanD = np.mean(d, axis=(0), dtype=np.float64)


# d = d.astype(float)
# d = (d - meanD)*128.0
# dtst = (dtst - meanD)*128.0
# m = np.amax(d)

print("train: %s seconds ---" % (time.time() - start_time))
create_lmdb(d2, l2, './data/', 'train', '24x24', meanD)



# meanX = np.mean(X, axis=(0), dtype=np.float64)
# devX = np.std(X, axis=(0), dtype=np.float64) + 0.00000000001

print("--- %s seconds ---" % (time.time() - start_time))
