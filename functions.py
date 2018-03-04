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
