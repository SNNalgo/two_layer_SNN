import functions
import matplotlib.pyplot as plt
import numpy as np
import pickle

I = 4e-9*np.ones((1,1000))
t = 0.1/1000
Rp = 50*t

nv,spikes = functions.LIF(I,t,Rp)
n1 = 4
n2 = 3
spikes1,spikes2 = functions.init_spike_times(n1,n2)

plt.figure()
plt.plot(t*np.array(range(1000)),nv.T)
plt.figure()
plt.plot(t*np.array(range(1000)),spikes.T)
plt.show()
'''
pickle_file = open('iris_data_1.pickle','rb')
load = pickle.load(pickle_file)

data = load['data']
targets = load['targets']
'''