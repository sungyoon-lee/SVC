# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:57:17 2017

@author: User
"""

import numpy as np
import time
import scipy.io as sio
import sklearn.metrics.pairwise as kernels
from scipy.linalg import inv
from scipy.sparse import csgraph as cg
from IPython import embed
from SVC_init import *

data_path='data/ring'
input = load_data(data_path)
print(input.shape)
print("with data",data_path)
support = "SVDD"
hyperparams = {'ker': 'rbf', 'arg': 0.5, 'C': 0.5}

#hyperparams = [100*np.ones((input.shape[0],1)), 1, 10]
st=time.time()
model = supportmodel(input,support, hyperparams)
et=time.time()
print("Training time:", et-st)
print("---------------------------------")
labmodel = labeling(model,"S-MSC")
#labmodel.run() 
et1=time.time()
print("Labeling time:", et1-et)
print("---------------------------------")
print(labmodel.cluster_label)
embed()
