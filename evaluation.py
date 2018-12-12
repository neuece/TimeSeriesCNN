import numpy as np
import pandas as pd
from model import Casual_CNN_Model

def evaluate_ts(input_ts, predict_len, lr):
    # input_ts 1d array of time series
    # predict_len: output ts length 
    # lr:learning rate
    
    input_ts = input_ts[~pd.isna(input_ts)]
    length = len(input_ts)-1

    input_ts = np.atleast_2d(np.asarray(input_ts))

    model = Casual_CNN_Model(length, lr)    
    model.summary()

    X = input_ts[:-1].reshape(1,length,1)
    y = input_ts[1:].reshape(1,length,1)
    
    model.fit(X, y, epochs=3000)
    
    pred_array = np.zeros(predict_len).reshape(1,predict_len,1)
    X_test_initial = input_ts[1:].reshape(1,length,1)

    pred_array[:,0,:] = model.predict(X_test_initial)[:,-1:,:]
    for i in range(predict_len-1):
        pred_array[:,i+1:,:] = model.predict(np.append(X_test_initial[:,i+1:,:], 
                               pred_array[:,:i+1,:]).reshape(1,length,1))[:,-1:,:]
    
    return pred_array.flatten()