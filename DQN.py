import StringIO
import numpy as np
import os
os.environ["GLOG_minloglevel"] = "1"
import caffe
import socket
import sys
import threading
import random
import tempfile

D=[]
maxD=10000
minD=500
C=1000
sampleSize=32
batchSize=256#32*8
dim=24

def main():
    
    PORT = int(sys.argv[1])

    HOST = ""
    
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(5)
    while True:
        print("Waiting for connection")
        conn, addr = sock.accept()
        conn.setblocking(1) 

        th = threading.Thread(target=processRequests, args = (conn.makefile(),1))
        th.start()

    sock.close()

def processRequests(conn,bla):
    print("Thread started")
    global D
    D=[]
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('DQNSolver.prototxt')
    #solver.net.copy_from("../microrts/data/caffe/"+str(dim)+"x"+str(dim)+".caffemodel")
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.close()
    solver.test_nets[0].save(f.name)
    targetNet = caffe.Net('DQN.prototxt', caffe.TEST, weights=f.name)
    
    it=0
    while True:
        msgType = conn.readline()
        #print msgType 
        if len(msgType)==0:
            conn.close()
            return

        if msgType=="eval\n":
            data = conn.readline()
            data += conn.readline()
            #print data 
            data = map(int, data.split())
            value=evaluate(data, solver.test_nets[0])
            conn.write(str(value)+"\n")
            conn.flush()
    	elif msgType=="store\n":
            data = conn.readline()
            #print data 
            w, l, planes = map(int, data.split())    
            size=w*l*planes
            #assert size==dim*dim*25
            data = conn.readline()
            #print data 
            data = map(int, data.split())
            plane_data = np.zeros(size)
            plane_data[data] = 1
            current = plane_data
            data = conn.readline()
            #print data 
            action = int(data)
            data = conn.readline()
            #print data 
            reward = float(data)
            data = conn.readline()
            #print data 
            w, l, planes = map(int, data.split())    
            size=w*l*planes
            #assert size==dim*dim*25
            data = conn.readline()
            #print data 
            data = map(int, data.split())
            plane_data = np.zeros(size)
            plane_data[data] = 1
            nextState = plane_data
            data = conn.readline()
            #print data 
            terminal = bool(int(data))
            D.append((current, action, reward, nextState, terminal))
            #print D
            if len(D)>=minD:
                #print "Min queue filled at it: "+str(it)
                if it % C == 0:
        	    solver.test_nets[0].save(f.name)
        	    targetNet = caffe.Net('DQN.prototxt', caffe.TEST, weights=f.name)
    	        train(solver, targetNet,it)
            it+=1
        else:
            print("Wrong message type: "+msgType)
            assert False

def train(solver, targetNet, it):
    global D
    l=len(D)
    if l>maxD:
        D=D[l-maxD:]
    #indices = np.random.randint(0,len(D),sampleSize)
    #miniBatch = [D[i] for i in indices]
    #miniBatch = D[:sampleSize]
    miniBatch = random.sample(D,sampleSize)

    #for i in range(sampleSize):
    #	assert len(miniBatch[i][3])==dim*dim*25

    
    #set labels:either r_j or r_j+l*targetNet eval
    plane_data = [entry[3] for entry in miniBatch]
    #print plane_data
    #print plane_data.shape
    x = np.reshape(plane_data, [sampleSize,25,dim,dim])
    x = rotations(x)
    #print x.shape
    targetNet.blobs['data'].reshape(batchSize,25,dim,dim)
    targetNet.blobs['data'].data[...] =  x
    out = targetNet.forward()
    #print np.mean(np.argmax(out['q_values'],1))
    #if it%C==0:
    #    print out['q_values'][:,:,0,0]
        #print np.argmax(out['q_values'],1)
    q=out['q_values'][:,:,0,0]

    rewards = [entry[2] for entry in miniBatch]
    terminal = [entry[4] for entry in miniBatch]
    actions = [entry[1] for entry in miniBatch]
    filters = np.zeros((batchSize,4,1,1))
    y = np.zeros((batchSize,4,1,1))
    for i in range(0,batchSize):
        filters[i][actions[i//8]]=1
        y[i][actions[i//8]]=rewards[i//8]
        if not terminal[i//8]:
            assert rewards[i//8]==0
            y[i][actions[i//8]]=np.clip(0.99*q[i].max(),-1.0,1.0)

    #print filters
    #print rewards
    #print y
    #set data blob on solver
    plane_data = [entry[0] for entry in miniBatch]
    x = np.reshape(plane_data, [sampleSize,25,dim,dim])
    x = rotations(x)
    solver.net.blobs['data'].data[...] =  x
    solver.net.blobs['reward'].data[...] =  y
    solver.net.blobs['filter'].data[...] =  filters

    solver.step(1)
    #print solver.net.blobs['q_values'].data
    #every C steps, targetNet=solver.net
	

def evaluate(data, net):
    w, l, planes = data[0:3]
    
    size=w*l*planes
    plane_data = np.zeros(size)
    plane_data[data[3:len(data)]] = 1

    x = np.reshape(plane_data, [1, planes, w, l])
    net.blobs['data'].reshape(1,planes,w,l)
    net.blobs['data'].data[...] =  x

    # compute
    out = net.forward()
    print str(out['q_values'][:,:,0,0])+"\t\t"+str(out['q_values'][0,:,0,0].argmax())
    return out['q_values'][0,:,0,0].argmax()

def rotations(data):
    aug_data = np.zeros((batchSize, data[0].shape[0], data[0].shape[1], data[0].shape[2]), dtype=np.uint8)

    for idx, array in enumerate(data):
        for i in xrange(array.shape[0]):
            aug_data[idx*8+0][i] = array[i]
            aug_data[idx*8+1][i] = np.rot90(array[i])
            aug_data[idx*8+2][i] = np.rot90(array[i], 2)
            aug_data[idx*8+3][i] = np.rot90(array[i], 3)
            aug_data[idx*8+4][i] = np.flipud(aug_data[idx*8+0][i])
            aug_data[idx*8+5][i] = np.fliplr(aug_data[idx*8+1][i])
            aug_data[idx*8+6][i] = np.flipud(aug_data[idx*8+2][i])
            aug_data[idx*8+7][i] = np.fliplr(aug_data[idx*8+3][i])
    return aug_data


main()
