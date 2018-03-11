import functions
import matplotlib.pyplot as plt
import numpy as np
import pickle

input_file = open('iris_data_1.pickle','rb')
load = pickle.load(input_file)

data = load['data']
targets = load['targets']
targets = np.int32(targets)

g = 30e-9
c = 300e-12
vt = 20e-3
el = -70e-3
# data - (numSamples, numInputNeurons)
numSamples = data.shape[0]
n2 = 3  #no. of level 2 neurons
n1 = data.shape[1]  #no. of level 1 neurons
Imax = 4e-9

T = 100e-3
dt = 0.2e-3
m = np.int32(T/dt)
rp = 5e-3

I0 = 10e-12
I0n = 1e-10
tauM = 10e-3
tauM2 = 50e-3
tauS = tauM/4

gammaUp = 1.7
gammaDn = -2.7
tauUp = 10e-3
tauDn = 2*tauUp
mu = 1.7

w1 = np.random.randn(n1,n2)*20+100
w2 = np.random.randn(n2,n2)*100-8000

wmax = 300
wmin = 0

for i in range(n2):
    w2[i,i] = 0
I = Imax*data.T

#generate all the first level responses
level1_spikes = np.zeros((n1,m,numSamples))
for i in range(numSamples):
    I_level1 = I[:,i:i+1]*np.ones((1,m))
    nv1,spikes1 = functions.LIF(I_level1,dt,rp)
    level1_spikes[:,:,i] = spikes1
s2 = np.zeros((n2,m,numSamples))
v2 = np.zeros((n2,m,numSamples))

Ib = 20e-9
bus2 = -10000*np.ones((n2,1))
success = []
diff = []
prev_w1 = w1
num_train_samples = 45
loop = 1
epoch = 0
max_epoch =  2
while loop>0:
    #one showing of the data
    for i in range(num_train_samples):
        s1 = level1_spikes[:,:,i]
        s2[:,:,i] = np.zeros((n2,m))
        v2[:,0,i] = el
        Isyn12 = np.zeros((n2,m))#current 1st to 2nd layer
        Isyn22 = np.zeros((n2,m))#lateral current
        Iapp2 = -Ib*np.ones((n2,m)) #applied bias current
        Iapp2[targets[i],:] = 0
        bus1 = -10000*np.ones((n1,1))
        bus2 = bus2-T
        #spike_times1 = {}
        #spike_times2 = {}
        spike_times1,spike_times2 = functions.init_spike_times(n1,n2)
        for j in range(1,m):
            t = np.float64(j)*dt
            bus1[s1[:,j]>0]=t
            
            #simulate voltages and currents
            k1 = (1/c)*(-g*(v2[:,j-1,i]-el*np.ones((n2,1))) + Isyn12[:,j-1]+Isyn22[:,j-1]+Iapp2[:,j-1])
            Vph = v2[:,j-1,i] + dt*k1
            k2 = (1/c)*(-g*(Vph-el*np.ones((n2,1))) + Isyn12[:,j-1]+Isyn22[:,j-1]+Iapp2[:,j])
            Vnew = v2[:,j-1,i] + dt*((k1+k2)/2)
            v2[:,j,i] = Vnew
            for k in range(n1):
                if (s1[k,j]>0):
                    spike_times1[k].append(t)
                    dw_dn = gammaDn*(np.pow((w1[k,:]/wmax),mu))*((np.exp((bus2-t)/tauDn)).T)
                    w1[k,:] = w1[k,:]+dw_dn
                if (len(spike_times1[k])>0):
                    np_times = np.array(spike_times1[k])
                    Isyn12[:,j] = Isyn12[:,j]+I0*(w1[k,:].T)*(np.sum(np.exp((np_times-t)/tauM)-np.exp((np_times-t)/tauS)))
            for k in range(n2):
                if (Vnew[k]>vt):
                    spike_times2[k].append(t)
                    bus2[k,0] = t
                    v2[k,j-1,i] = 0.1
                    s2[k,j,i] = 1
                    v2[k,j,i] = el
                    dw1 = gammaUp*(np.pow((1-w1[:,k]/wmax),mu))*(np.exp((bus1-t)/tauUp))
                    w1[:,k] = w1[:,k]+dw1
                if (len(spike_times2[k])>0):
                    np_times = np.array(spike_times2[k])
                    Isyn22[:,j] = Isyn22[:,j]+I0n*(w2[k,:].T)*(np.sum(np.exp((np_times-t)/tauM2)))
            w1[w1>wmax] = wmax
            w1[w1<wmin] = wmin
    epoch = epoch+1
    if (epoch%5 == 4):
        success = functions.test_iris(w1,w2,level1_spikes,targets)
        print("success % is : ",success)
    if (epoch == max_epoch):
        loop = 0
