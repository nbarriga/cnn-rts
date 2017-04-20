import StringIO
import numpy as np
import os
os.environ["GLOG_minloglevel"] = "1"
import caffe
import socket
import sys
import multiprocessing as mp
import random
import tempfile

D=[]
maxD=10000
minD=500
C=1000
batchSize=256
dim=128
channels=26

def main():
    
    PORT = int(sys.argv[1])

    HOST = ""
    
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(5)
    while True:
        print("Waiting for connection")
        conn, addr = sock.accept()
        conn.setblocking(1) 

        th = mp.Process(target=processRequests, args = (conn.makefile(),1))
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
    #f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    #f.close()
    fname='snapshots/DQN-iter-0.caffemodel'
    solver.test_nets[0].save(fname)
    targetNet = caffe.Net('DQN.prototxt', caffe.TEST, weights=fname)
    
    it=0
    while True:

	#current state
	current = getPlanesFlat(conn)

	#action
        data = conn.readline()
        #print data 
        action = int(data)

	#reward
        data = conn.readline()
        #print data 
        reward = float(data)

	#next state
        nextState = getPlanesFlat(conn)

	#terminal
        data = conn.readline()
        #print data 
        terminal = bool(int(data))
        D.append((current, action, reward, nextState, terminal))
        #print D
        if len(D)>=minD:
            #print "Min queue filled at it: "+str(it)
            if it % C == 0:
                fname='snapshots/DQN-iter-'+str(it)+'.caffemodel'
                solver.test_nets[0].save(fname)
                targetNet = caffe.Net('DQN.prototxt', caffe.TEST, weights=fname)
    	    train(solver, targetNet,it)
            solver.test_nets[0].save('snapshots/DQN-current.caffemodel')
        it+=1
    except Exception as ex:
        print str(ex)
    finally:
        conn.close()

def getPlanes(conn):
    header = conn.readline()
    #print header

    if len(header)==0:
        conn.close()
        raise Exception("Empty header!")

    #current state
    header = map(int, header.split())
    w, l, planes, one_hots = header[0],header[1],header[2],header[3:]
    size=w*l*planes
    assert size==dim*dim*channels
    plane_data = np.zeros(size)
    indices =  conn.readline().split()
    #print indices
    plane_data[map(int, indices)] = 1
    x = np.reshape(plane_data, [planes, w, l])
    for index in one_hots:
        x[index]=1
    return x

def getPlanesFlat(conn):
    x = getPlanes(conn)
    return np.reshape(x, [dim*dim*channels])


def train(solver, targetNet, it):
    global D
    l=len(D)
    if l>maxD:
        D=D[l-maxD:]
    #indices = np.random.randint(0,len(D),sampleSize)
    #miniBatch = [D[i] for i in indices]
    #miniBatch = D[:sampleSize]
    miniBatch = random.sample(D,batchSize)

    #for i in range(sampleSize):
    #	assert len(miniBatch[i][3])==dim*dim*25

    
    #set labels:either r_j or r_j+l*targetNet eval
    plane_data = [entry[3] for entry in miniBatch]
    #print plane_data
    #print plane_data.shape
    x = np.reshape(plane_data, [batchSize,channels,dim,dim])
    rot = np.random.randint(0,8,(batchSize))
    for i in range(0,batchSize):
        data[i] = self.rotate(data[i],rot[i])
    #x = rotations(x)
    #print x.shape
    targetNet.blobs['data'].reshape(batchSize,channels,dim,dim)
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
        filters[i][actions[i]]=1
        y[i][actions[i]]=rewards[i]
        if not terminal[i]:
            assert rewards[i]==0
            y[i][actions[i]]=np.clip(0.99*q[i].max(),-1.0,1.0)

    #print filters
    #print rewards
    #print y
    #set data blob on solver
    plane_data = [entry[0] for entry in miniBatch]
    x = np.reshape(plane_data, [batchSize,channels,dim,dim])
    #x = rotations(x)
    
    for i in range(0,batchSize):
        data[i] = self.rotate(data[i],rot[i])

    solver.net.blobs['data'].data[...] =  x
    solver.net.blobs['reward'].data[...] =  y
    solver.net.blobs['filter'].data[...] =  filters

    solver.step(1)
    #print solver.net.blobs['q_values'].data
    #every C steps, targetNet=solver.net
	

#def evaluate(data, net):
#    w, l, planes = data[0:3]
#    
#    size=w*l*planes
#    plane_data = np.zeros(size)
#    plane_data[data[3:len(data)]] = 1
#
#    x = np.reshape(plane_data, [1, planes, w, l])
#    net.blobs['data'].reshape(1,planes,w,l)
#    net.blobs['data'].data[...] =  x
#
#    # compute
#    out = net.forward()
#    print str(out['q_values'][:,:,0,0])+"\t\t"+str(out['q_values'][0,:,0,0].argmax())
#    return out['q_values'][0,:,0,0].argmax()
#
#def rotations(data):
#    aug_data = np.zeros((batchSize, data[0].shape[0], data[0].shape[1], data[0].shape[2]), dtype=np.uint8)
#
#    for idx, array in enumerate(data):
#        for i in xrange(array.shape[0]):
#            aug_data[idx*8+0][i] = array[i]
#            aug_data[idx*8+1][i] = np.rot90(array[i])
#            aug_data[idx*8+2][i] = np.rot90(array[i], 2)
#            aug_data[idx*8+3][i] = np.rot90(array[i], 3)
#            aug_data[idx*8+4][i] = np.flipud(aug_data[idx*8+0][i])
#            aug_data[idx*8+5][i] = np.fliplr(aug_data[idx*8+1][i])
#            aug_data[idx*8+6][i] = np.flipud(aug_data[idx*8+2][i])
#            aug_data[idx*8+7][i] = np.fliplr(aug_data[idx*8+3][i])
#    return aug_data

  def rotate(data, r):
    for i in xrange(data.shape[0]):
      if r==0:
        pass
      elif r==1:
        data[i] = np.rot90(data[i])
      elif r==2:
        data[i] = np.rot90(data[i], 2)
      elif r==3:
        data[i] = np.rot90(data[i], 3)
      elif r==4:
        data[i] = np.flipud(data[i])
      elif r==5:
        data[i] = np.rot90(data[i])
        data[i] = np.fliplr(data[i])
      elif r==6:
        data[i] = np.rot90(data[i], 2)
        data[i] = np.flipud(data[i])
      elif r==7:
        data[i] = np.rot90(data[i], 3)
        data[i] = np.fliplr(data[i])
      else:
        raise Exception('Wrong rotation')
    return data

main()
