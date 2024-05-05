import numpy as np

def kalman_smoother(model, data, x0, P0):
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

    # print(data)
    T = data.shape[1]

    # init estimates
    D = x0.size
    # print(data.shape)
    # print(x0.shape)
    x = np.zeros((D, T))
    P = np.zeros((D, D, T))
    P_pred = np.zeros((D, D, T))
    x_pred = np.zeros((D, T))
    print(x_pred.shape)

    #unpack alphabet
    A = model['A']
    C = model['C']
    Q = model['Q']
    R = model['R']

    #forward pass
    for t in range(T):

        #Prediction Step
        if t > 0:
            # Prediction mean
            x_pred[:, t] = np.matmul(A,x[:, t-1])
            # Prediction covariance
            P_pred[:,:, t] = Q + np.matmul(np.matmul(A,P[:,:,t-1]),A.T)
        else:
            # Initialize predicted mean / var to prior
            x_pred[:, t] = x0
            P_pred[:,:, t] = P0

        # Measurement update
        z_i = data[:, t]
        # Kalman Gain Matrix
        # print(z_i)
        K = np.matmul(P_pred[:,:, t], np.matmul(C.T, np.linalg.inv(np.matmul(np.matmul(C,P_pred[:,:, t]),C.T) + R)))
        # Filter Mean
        x[:, t] = x_pred[:,t] + np.matmul(K,(z_i - np.matmul(C,x_pred[:,t])))
        # Filter Covariance
        P[:,:, t] = P_pred[:,:,t] - np.matmul(K,np.matmul(C,P_pred[:,:,t]))

    Xf = x
    Pf = P

    # reuse moments
    x = np.zeros((D, T))
    P = np.zeros((D, D, T))

    # backward pass
    x[:, -1] = Xf[:, -1]
    P[:,:, -1] = Pf[:,:, -1]
    for t in range((T-2),-1, -1):
        C = np.matmul(np.matmul(Pf[:,:, t],A.T) ,np.linalg.inv(P_pred[:,:,t+1]))
        x[:, t] = Xf[:, t] + np.matmul(C, (x[:, t+1] - x_pred[:, t+1] ))
        P[:,:, t] = Pf[:,:, t] + np.matmul(np.matmul(C, (P[:,:, t+1] - P_pred[:,:, t+1] )) , C.T)

    Xs = x
    Ps = P

    return Xf, Pf, Xs, Ps
