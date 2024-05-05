import numpy as np
from scipy.stats import multivariate_normal

def particle_filter(nParticles, model, data, x0, P0, resample=True):
    '''
    Inputs:
        model: the data model structure, containing the fields A,C,Q,R, as
                defined in the handout
        data: time-series data (LxT)
        x0: initial guess for the state (D-array)
        P0: covariance matrix on the initial state (DxD)
    Outputs:
        X: D x T array containing the D-dimensional posterior mean of the
            estimate for the states from time t=1 to time=T
        P: D x D x T array containing the covariances of particles from time
            t=1 to time=T
    '''

    D = len(x0)
    L = data.shape[0]
    T = data.shape[1]
  
    # Init stuff
    X = np.zeros((D,T))
    P = np.zeros((D, D, T));
    W = np.zeros((nParticles, T))
    x = np.zeros((nParticles, D, T))
    
    # unpack alphabet
    A, C, Q, R = model['A'], model['C'], model['Q'], model['R']

    # sample initial particles
    x_sim = np.random.multivariate_normal(x0, P0, size=nParticles)
    log_w_sim = multivariate_normal.logpdf(data[:,0].T - x_sim@C.T, mean=np.zeros(L), cov=R)    

    # time recursion
    for t in range(1,T):
        
        # normalize weights
        log_w_sim = log_w_sim - max( log_w_sim )
        W[:,t-1] = np.exp(log_w_sim) / sum(np.exp(log_w_sim))
        assert(not np.any(np.isnan(log_w_sim)))
        
        # resample particles    
        if resample:
            I = np.random.choice( nParticles, nParticles, True, W[:,t-1])            
            x[:,:,t-1] = x_sim[ I, : ]
            W[:,t-1] = 1 / nParticles        
        else:
            x[:,:,t-1] = x_sim.copy()

        # compute moments
        X[:,t-1] = x[:,:,t-1].T @ W[:,t-1]
        res = x[:,:,t-1] - X[:,t-1].T
        Psum = 0
        for i in range(nParticles):
            Psum += W[i,t-1] * np.outer(res[i,:], res[i,:])
        P[:,:,t-1] = Psum
        
        # propogate
        mu = x[:,:,t-1] @ A.T;    
        x_sim = mu + np.random.multivariate_normal(np.zeros(D), Q, nParticles)        
        
        # compute weights            
        llhood = multivariate_normal.logpdf( data[:,t].T - x_sim@C.T, mean=np.zeros(L), cov=R)          
        log_w_sim = np.log( W[:,t-1] ) + llhood
  
    # compute final moments  
    x[:,:,T-1] = x_sim
    log_w_sim = log_w_sim - max( log_w_sim )
    W[:,T-1] = np.exp(log_w_sim) / sum( np.exp(log_w_sim) )
    X[:,T-1] = x[:,:,T-1].T @ W[:,T-1]
    res = x[:,:,T-1] - X[:,T-1].T
    Psum = 0
    for i in range(nParticles):
            Psum += W[i,T-1] * np.outer(res[i,:], res[i,:])
    P[:,:,T-1] = Psum

    return X, P
  
    