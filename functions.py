import numpy as np

def LIF(I, t, Rp):
    n,m = I.shape
    g = 30e-9
    C = 300e-12
    Vt = 20e-3
    El = -70e-3
    nv = El*np.ones(I.shape)
    refractoryCount = np.zeros((n,1))
    spikes = np.zeros(I.shape)
    for i in range(1,m):
        temp_spikes = np.zeros((n,1))
        k1 = (1/C)*(-g*(nv[:,i-1:i]-El)+I[:,i-1:i])
        Vph = nv[:,i-1:i]+t*k1
        k2 = (1/C)*(-g*(Vph-El)+I[:,i:i+1])
        Vnew = nv[:,i-1:i] + t*((k1+k2)/2)
        temp_spikes[Vnew>Vt] = 1
        refractoryCount[Vnew>Vt] = Rp+t
        Vnew[Vnew>Vt] = El
        Vnew[refractoryCount>0] = El
        refractoryCount[refractoryCount>0] = refractoryCount[refractoryCount>0] - t
        spikes[:,i:i+1] = temp_spikes
        nv[:,i:i+1] = Vnew
    return nv,spikes

def init_spike_times(n1,n2):
    spikes1 = {}
    spikes2 = {}
    for i in range(n1):
        spikes1[i]=[]
    for i in range(n2):
        spikes2[i]=[]
    return spikes1,spikes2

def test_iris(w1,w2,level1_spikes,targets):
    success = 0
    g = 30e-9
    c = 300e-12
    vt = 20e-3
    el = -70e-3
    dt = 0.2e-3
    I0 = 10e-12
    I0n = 1e-10
    tauM = 10e-3
    tauS = tauM/4
    numSamples = level1_spikes.shape[2]
    n1 = w1.shape[0]
    n2 = w1.shape[1]
    m = level1_spikes.shape[1]
    s2 = np.zeros((n2,m,numSamples))
    v2 = np.zeros((n2,m,numSamples))
    for i in range(numSamples):
        s1 = level1_spikes[:,:,i]
        s2[:,:,i] = np.zeros((n2,m))
        v2[:,0,i] = el
        Isyn12 = np.zeros((n2,m))#current 1st to 2nd layer
        Isyn22 = np.zeros((n2,m))#lateral current
        #spike_times1 = {}
        #spike_times2 = {}
        spike_times1,spike_times2 = init_spike_times(n1,n2)
        for j in range(1,m):
            t = np.float64(j)*dt
            #simulate voltages and currents
            k1 = (1/c)*(-g*(v2[:,j-1,i]-el) + Isyn12[:,j-1]+Isyn22[:,j-1])
            Vph = v2[:,j-1,i] + dt*k1
            k2 = (1/c)*(-g*(Vph-el) + Isyn12[:,j-1]+Isyn22[:,j-1])
            Vnew = v2[:,j-1,i] + dt*((k1+k2)/2)
            v2[:,j,i] = Vnew
            for k in range(n1):
                if (s1[k,j]>0):
                    spike_times1[k].append(t)
                if (len(spike_times1[k])>0):
                    np_times = np.array(spike_times1[k])
                    Isyn12[:,j] = Isyn12[:,j]+I0*(w1[k,:].T)*(np.sum(np.exp((np_times-t)/tauM)-np.exp((np_times-t)/tauS)))
            for k in range(n2):
                if (Vnew[k]>vt):
                    spike_times2[k].append(t)
                    v2[k,j-1,i] = 0.1
                    s2[k,j,i] = 1
                    v2[k,j,i] = el
                if (len(spike_times2[k])>0):
                    np_times = np.array(spike_times2[k])
                    #Isyn22[:,j] = Isyn22[:,j]+I0n*(w2[k,:].T)*(np.sum(np.exp((np_times-t)/tauM2)))
                    Isyn22[:,j] = Isyn22[:,j]+I0n*(w2[k,:].T)
        correct_count = 0
        for j in range(n2):
            if ((j==targets[i] and len(spike_times2[j])>0) or (j!=targets[i] and len(spike_times2[j])==0)):
                correct_count = correct_count + 1
        if (correct_count == n2):
            success = success+1
    return (100.0*success)/numSamples