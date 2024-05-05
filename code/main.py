import numpy as np
import matplotlib.pyplot as plt
import h5py

# local stuff
from particle_filter import particle_filter
from kalman_smoother import kalman_smoother
from plot_truth import plot_truth

seed = 0
np.random.seed(seed)

with h5py.File('./track.mat', 'r') as f:
    # print(f.keys())
    x0 = np.squeeze( np.array(f['x0'][()]) )
    P0 = np.array(f['P0'][()])
    model = {}
    model['A'] = np.array(f['model']['A'][()]).T
    model['C'] = np.array(f['model']['C'][()]).T
    model['Q'] = np.array(f['model']['Q'][()])
    model['R'] = np.array(f['model']['R'][()])
    states = np.array(f['states'][()])
    data = np.array(f['data'][()]) # D x T (D=1 here)

# Model is a data structure that contains fields: A,Q,R,C, which correspond
# to the parameters defined in Problem 1 of the handout.
T = data.shape[1]
plot_truth(states, data)


# Run and plot Kalman filter
Xf, Pf, Xs, Ps = kalman_smoother(model, data, x0, P0)
E = 2 * np.sqrt(Pf[0,0,:])
plt.errorbar(np.arange(T), Xf[0,:], E, color='c', label='Kalman Filter')

# Run and plot particle filter (with resampling)
numRuns = 3
nParticles = 1000
for i in range(numRuns):
    X_pf, P_pf = particle_filter(nParticles, model, data, x0, P0)
    if i==0:
        plt.plot( range(T), X_pf[0,:], '.-m', label='Particle Filter') 
    else:
        plt.plot( range(T), X_pf[0,:], '.-m') 

# adjust plot
plt.xlim([0, T+1])
plt.legend()

# corrupt data
to_flip = np.random.rand(T, 1) < 0.1
to_flip = to_flip.T
noisy_vals = 40 * np.random.normal(0, 1, size=T)
noisy_vals = noisy_vals[np.newaxis, :]
noisy_data = (1-to_flip) * data + to_flip * noisy_vals

# run and plot Kalman
plt.figure()
plot_truth(states, noisy_data)
Xf_noisy, Pf_noisy, Xs_noisy, Ps_noisy = \
    kalman_smoother(model, noisy_data, x0, P0)
plt.errorbar(np.arange(T), Xf_noisy[0,:], E, color='c', label='Kalman Filter')

# run and plot particle filter
for i in range(numRuns):
    X_pf, P_pf = particle_filter(nParticles, model, noisy_data, x0, P0)
    if i==0:
        plt.plot( range(T), X_pf[0,:], '.-m', label='Particle Filter') 
    else:
        plt.plot( range(T), X_pf[0,:], '.-m') 

# adjust plot
plt.title('Noisy Data')
plt.xlim([0, T+1])
plt.legend()

# Run and plot particle filter (without resampling)
plt.figure()
plot_truth(states, data)
plt.errorbar(np.arange(T), Xf[0,:], E, color='c', label='Kalman Filter')
for i in range(numRuns):
    X_pf, P_pf = particle_filter(nParticles, model, data, x0, P0, resample=False)
    if i==0:
        plt.plot( range(T), X_pf[0,:], '.-m', label='Particle Filter') 
    else:
        plt.plot( range(T), X_pf[0,:], '.-m') 

# adjust plot
plt.title('No Resampling')
plt.xlim([0, T+1])
plt.legend()

plt.show()

