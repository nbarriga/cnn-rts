#!/usr/bin/env python 
import sys
sys.path.append("~/git-working/caffe/python/")

import numpy as np
import lmdb
import caffe
import time
import math
import shutil
import h5py

start_time = time.time()
#np.random.seed(24)

def read_games(channels, dimension, nrGames, path, process_labels, startGame, datatype):
    data = np.zeros((nrGames, channels, dimension, dimension), dtype=np.uint8)
    #labels = np.zeros(nrGames, dtype=np.uint8)
    labels = np.zeros(nrGames, dtype=datatype)

    for i in range(nrGames):
        if i % 1000 == 0:
            print i

        #array = np.fromfile(path+str(startGame+i), dtype=np.uint8, count=-1, sep=" ")
        array = np.fromfile(path+str(startGame+i), dtype=np.uint8, count=(3+channels*dimension*dimension), sep=" ")
        w = array[0]
        h = array[1]
        c = array[2]

        assert w == h and w == dimension and c == channels

        f = open(path+str(startGame+i),'r')
        f.seek(-40,2)
        if datatype == np.uint8:
            labels[i] = int(f.readlines()[-1])
        else:
            labels[i] = float(f.readlines()[-1])

        #convert [0, 1] to [-1, 1]
        #labels[i]=(float(f.readlines()[-1])+1)/2.0
        #labels[i] = array[-1]

#        if process_labels:
#            t = int(labels[i])/channels
#            if t == 2:
#                labels[i] = 2  # draw
#            elif t < 2:
#                labels[i] = 0  # defeat
#            else:
#                labels[i] = 1  # victory

        #array = np.delete(array, array.size-1)
        array = np.delete(array, [0, 1, 2])
        # print array.size
        data[i] = np.reshape(array, [c, w, h])
    return data, labels

def create_db(data, labels, save_path, name, extra, mean, batch, batch_size, delete, backend):
    if backend == "lmdb":
        create_lmdb(data, labels, save_path, name, extra, mean, batch, batch_size, delete)
    elif backend == "hdf5":
        create_hdf5(data, labels, save_path, name, extra, mean, batch, batch_size, delete)
    else:
        print "Error: backend "+backend+" not supported."
        sys.exit()

def create_hdf5(data, labels, save_path, name, extra, mean, batch, batch_size, delete):

    listname = save_path + name + extra + '_list.txt'
    dbname = save_path + name + extra + '_' + str(batch) + '.h5'

    print "deleting hdf5"
    try:
        os.remove(dbname)
    except:
        print "error deleting db "+str(batch)
    if delete:
        try:
            os.remove(listname)
        except:
            print "error deleting db list"

    with h5py.File(dbname,'w') as H:
        H.create_dataset( 'data', data=data ) # note the name X given to the dataset!
        H.create_dataset( 'label', data=labels ) # note the name y given to the dataset!
    with open(listname,'a') as L:
        L.write( dbname+'\n' ) # list all h5 files you are going to use

    return

# save_path = '/media/root/EXTERNAL_2GB/'
def create_lmdb(data, labels, save_path, name, extra, mean, batch, batch_size, delete):
    buffer_size = 100
    DB_KEY_FORMAT = "{:0>10d}"

    #map_size = data[0].astype(float).nbytes * 2 * len(labels) + 100
    map_size= 1000000000000
    #print "Map size: "+str(map_size/1000000)+"MB"
    lmdb_name = save_path + name + extra

    c = data[0].shape[0]
    h = data[0].shape[1]
    w = data[0].shape[2]

    start_idx=batch*batch_size*8
    #print "idx: "+str(start_idx)
    try:
        if delete:
            print "deleting lmdb"
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

                str_id = DB_KEY_FORMAT.format(start_idx + buffer_size*idx + current_idx)
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


def augment(data, labels, datatype):

    aug_data = np.zeros((len(labels)*8, data[0].shape[0], data[0].shape[1], data[0].shape[2]), dtype=np.uint8)
    aug_labels = np.zeros((len(labels)*8), dtype=datatype)

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

#testGames = 3024
#trainGames = 26400
#samples = 12
meanD = 0
#dim = 128
#dimstr = str(dim)+'x'+str(dim)

dim = int(sys.argv[1])
dimstr = str(dim)+'x'+str(dim)
planes = int(sys.argv[2])
trainDirectory = sys.argv[3]
testDirectory = sys.argv[4]
outDirectory = sys.argv[5]

batch = int(sys.argv[6])
backend = sys.argv[7]
if sys.argv[8]=="float":
    datatype = np.float32
elif sys.argv[8]=="int":
    datatype = np.uint8
else:
    print "Error: unsupport data type: "+datatype
    sys.exit()

if backend == "lmdb" and datatype==np.float32:
    print "Error: LMDB doesn't support "+datatype
    sys.exit()


import fnmatch
import os

trainGames = len(fnmatch.filter(os.listdir(trainDirectory), 'game*'))
testGames = len(fnmatch.filter(os.listdir(testDirectory), 'game*'))

print("Reading "+str(testGames)+" test games")
d, l = read_games(planes, dim, testGames, testDirectory+'/game', 0, 0, datatype)
print("Reading done: %s seconds ---" % (time.time() - start_time))

print("Shuffling games")
dtst, ltst = shuffle(d, l)
d=[]
l=[]
print("Shuffling done: %s seconds ---" % (time.time() - start_time))

print("Creating test DB")
create_db(dtst, ltst, outDirectory, '/test', dimstr, meanD, 0, testGames, True, backend)
dtst = []
ltst = []
print("DB done: %s seconds ---" % (time.time() - start_time))

#batch = 2000
startGame = 0
delete = True;
i = 0
if batch > 0:
    while startGame+batch<trainGames+batch:
        print("Reading "+str(batch if startGame+batch<=trainGames else trainGames-startGame)+" training games")
        d, l = read_games(planes, dim, batch if startGame+batch<=trainGames else trainGames-startGame, trainDirectory+'/game', 0, startGame, datatype)
        print("Reading done: %s seconds ---" % (time.time() - start_time))

        print("Shuffling games")
        ds, ls = shuffle(d, l)
        d = []
        l = []
        print("Shuffling done: %s seconds ---" % (time.time() - start_time))

        print("Augmenting games")
        dsa, lsa = augment(ds, ls, datatype)
        ds = []
        ls = []
        print("Augmenting done: %s seconds ---" % (time.time() - start_time))
        
        print("Shuffling games")
        d2, l2 = shuffle(dsa, lsa)
        dsa = []
        lsa = []
        print("Shuffling done: %s seconds ---" % (time.time() - start_time))


        print("Creating training DB")
        create_db(d2, l2, outDirectory, '/train', dimstr, meanD, i, batch, delete, backend)
        d2 = []
        l2 = []
        print("DB done: %s seconds ---" % (time.time() - start_time))
        delete = False
        startGame+=batch
        i+=1
        print "progress: "+str(batch*i)+"/"+str(trainGames)+" samples"
else:
    print("Reading "+str(trainGames)+" training games")
    d, l = read_games(planes, dim, trainGames, trainDirectory+'/game', 0, 0, datatype)
    print("Reading done: %s seconds ---" % (time.time() - start_time))

    print("Shuffling games")
    ds, ls = shuffle(d, l)
    d = []
    l = []
    print("Shuffling done: %s seconds ---" % (time.time() - start_time))

    print("Augmenting games")
    dsa, lsa = augment(ds, ls, datatype)
    ds = []
    ls = []
    print("Augmenting done: %s seconds ---" % (time.time() - start_time))
        
    print("Shuffling games")
    d2, l2 = shuffle(dsa, lsa)
    dsa = []
    lsa = []
    print("Shuffling done: %s seconds ---" % (time.time() - start_time))


    print("Creating training DB")
    create_db(d2, l2, outDirectory, '/train', dimstr, meanD, 0, trainGames, True, backend)
    d2 = []
    l2 = []
    print("DB done: %s seconds ---" % (time.time() - start_time))

