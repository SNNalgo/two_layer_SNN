import numpy as np
from sklearn.datasets import load_iris
import random
import cPickle as pickle

data, target = load_iris(return_X_y=True)
print (data.shape)
print (target.shape)

maxVals = np.max(data, axis=0, keepdims=True)
minVals = np.min(data, axis=0, keepdims=True)
#print maxVals.shape

data = (data-minVals)/(maxVals-minVals)#scaling between 0 and 1

centers = [0.2, 0.5, 0.7]
sigma = 0.3
extended_data = np.zeros((data.shape[0],len(centers)*data.shape[1]))
for i in range(len(centers)):
    extended_data[:,i*data.shape[1]:(i+1)*data.shape[1]] = np.exp(-((data-centers[i])**2)/(2*(sigma**2)))
#print extended_data.shape
a = range(np.max(target)+1)
b = range(50)
random.shuffle(b)
print (a)
randomized_data = np.zeros(extended_data.shape)
randomized_targets = np.zeros(data.shape[0])
indices = np.zeros(data.shape[0])
offsets = [0,50,100]
for i in range(50):
    random.shuffle(a)
    randomized_targets[3*i]=a[0]
    randomized_data[3*i,:]=extended_data[b[i]+offsets[a[0]],:]
    indices[3*i]=b[i]+offsets[a[0]]
    
    randomized_targets[3*i+1]=a[1]
    randomized_data[3*i+1,:]=extended_data[b[i]+offsets[a[1]],:]
    indices[3*i+1]=b[i]+offsets[a[1]]
    
    randomized_targets[3*i+2]=a[2]
    randomized_data[3*i+2,:]=extended_data[b[i]+offsets[a[2]],:]
    indices[3*i+2]=b[i]+offsets[a[2]]

print(randomized_targets)
print(indices)
pickle_file = open('iris_data_1.pickle','wb')
save = {'data' : randomized_data, 'targets' : randomized_targets}
pickle.dump(save,pickle_file)
