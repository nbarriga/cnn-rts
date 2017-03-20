import sys, os
sys.path.insert(0,'/home/barriga/git-working/caffe/python')
import caffe
import numpy as np
import multiprocessing as mp

class PythonDataLayer(caffe.Layer):
  def setup(self,bottom,top):
    # === Read input parameters ===

    # params is a python dictionary with layer parameters.
    params = eval(self.param_str)

    # store input as class variables
    self.batch_size = params['batch_size']
    self.num_samples = params['num_samples']
    self.path = params['path']
    self.nextGame=0
    self.planes=params['channels'] 
    self.width=params['width']
    self.height=params['height']
    self.size=self.planes*self.width*self.height
    self.preload=params['preload']
    self.shuffle=params['shuffle']
    self.datatype=params['datatype']
    
    if self.datatype == int:
        self.dtype=np.uint8
    elif self.datatype == float:
        self.dtype=np.float32
    else:
        raise Exception('Label datatype must be int or float: '+str(self.datatype))

    if self.num_samples % self.batch_size != 0:
         raise Exception('number of samples must be a multiple of batch size')
    # === reshape tops ===
    # since we use a fixed input image size, we can shape the data layer
    # once. Else, we'd have to do it in the reshape call.
    # no "bottom"s for input layer
    if len(bottom)>0:
      raise Exception('cannot have bottoms for input layer')
    # make sure you have the right number of "top"s
    if len(top)!= 3:
      raise Exception('need three tops (data, data2, label) for input layer')

    top[0].reshape(self.batch_size, self.planes, self.width, self.height)
    top[1].reshape(self.batch_size, 16, 18, 18)
    # 2 label
    top[2].reshape(self.batch_size, 1)
    
    if self.preload == 1:
      self.alllines=[]
      for i in range(0,self.num_samples):
        f = open(self.path+str(i),'r')
        lines=f.readlines()
        self.alllines.append(lines)

    if self.shuffle == 1:
      self.games = np.random.permutation(self.num_samples)
    else:
      self.games = range(0,self.num_samples)

  def reshape(self,bottom,top):
    #constant data size
    pass

  def forward(self,bottom,top): 
    # do your magic here... feed **one** batch to `top`
    data = np.zeros((self.batch_size,self.planes,self.width,self.height))
    data2 = np.zeros((self.batch_size,16,18,18))
    labels = np.zeros((self.batch_size,1),self.dtype)
    for i in range(0,self.batch_size):
      if self.preload==1:
        lines=self.alllines[self.games[self.nextGame+i]]
      else:
        f = open(self.path+str(self.games[self.nextGame+i]),'r')
        lines=f.readlines()
      header = map(int,lines[0].split())
      w, l, p, one_hots = header[0],header[1],header[2],header[3:]
      if self.size != w*l*(p-16):
          raise Exception('file '+self.path+str(self.games[self.nextGame+i])+' with incorrect data dimensions')
      
      plane_data = np.zeros(self.size)
      plane_data[map(int, lines[1].split())] = 1

      data[i] = np.reshape(plane_data, [self.planes, self.width, self.height])

      for index in one_hots:
          data2[i,index-self.planes]=1

      labels[i] = self.dtype(lines[2])

    self.nextGame=(self.nextGame+self.batch_size)%self.num_samples

    top[0].data[...] = data 
    top[1].data[...] = data2 
    top[2].data[...] = labels

  def backward(self, top, propagate_down, bottom):
    # no back-prop for input layers
    pass
