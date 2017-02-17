import StringIO
import numpy as np
import caffe
import socket
import sys
import threading
import random

D=[]
batchSize=256

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
    caffe.set_device(1)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('DQNSolver.prototxt')
    targetNet = solver.test_nets[0] #change to a different network
    
    while True:
        msgType = conn.readline()
        print msgType 
        if len(msgType)==0:
            conn.close()
            return

        if msgType=="eval\n":
            data = conn.readline()
            data += conn.readline()
            print data 
            data = map(int, data.split())
            value=evaluate(data, targetNet)
            conn.write(str(value)+"\n")
            conn.flush()
    	elif msgType=="store\n":
            data = conn.readline()
            print data 
            w, l, planes = map(int, data.split())    
            size=w*l*planes
            data = conn.readline()
            print data 
            data = map(int, data.split())
            plane_data = np.zeros(size)
            plane_data[data] = 1
            current = plane_data
            #data = conn.recv(8192)
            data = conn.readline()
            print data 
            action = int(data)
            #data = conn.recv(8192)
            data = conn.readline()
            print data 
            reward = float(data)
            #data = conn.recv(8192)
            data = conn.readline()
            print data 
            w, l, planes = map(int, data.split())    
            size=w*l*planes
            data = conn.readline()
            print data 
            data = map(int, data.split())
            plane_data = np.zeros(size)
            plane_data[data] = 1
            nextState = plane_data
            #data = conn.recv(8192)
            data = conn.readline()
            print data 
            terminal = bool(int(data))
            D.append((current, action, reward, nextState, terminal))
            #print D
    	    train(solver, targetNet)
        else:
            assert False

def train(solver, targetNet):
    batchSize=1
    miniBatch = random.sample(D,batchSize)


    
    #set labels:either r_j or r_j+l*targetNet eval
    plane_data = [entry[3] for entry in miniBatch]
    x = np.reshape(plane_data, [batchSize,25,8,8])
    targetNet.blobs['data'].data[...] =  x
    out = targetNet.forward()
    print out['q_values'][0,:,0,0]
    q=out['q_values'][0,:,0,0]

    rewards = [entry[2] for entry in miniBatch]
    terminal = [entry[4] for entry in miniBatch]
    actions = [entry[1] for entry in miniBatch]
    filters = np.zeros((batchSize,4,1,1))
    y = np.zeros((batchSize,1,1,1))
    y[0,0,0,0]=1
    filters[:,2]=1
   # for i in range(0,batchSize):
   #     filters[i][actions[i]]=1
   #     y[i]=rewards[i]
   #     if not terminal[i]:
   #         assert rewards[i]==0
   #         y[i]+=0.99*q[i]

    #set data blob on solver
    plane_data = [entry[0] for entry in miniBatch]
    x = np.reshape(plane_data, [batchSize,25,8,8])
    solver.net.blobs['data'].data[...] =  x
    solver.net.blobs['reward'].data[...] =  y
    solver.net.blobs['filter'].data[...] =  filters

    solver.step(1)
    print solver.net.blobs['q_values'].data
    #every C steps, targetNet=solver.net
	

def evaluate(data, net):
    w, l, planes = data[0:3]
    
    size=w*l*planes
    plane_data = np.zeros(size)
    plane_data[data[3:len(data)]] = 1

    x = np.reshape(plane_data, [planes, w, l])
    net.blobs['data'].data[...] =  x

    # compute
    out = net.forward()
    print out['q_values'][0,:,0,0]
    return out['q_values'][0,:,0,0].argmax()

main()
