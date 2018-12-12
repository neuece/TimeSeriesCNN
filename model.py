# keras implementation of WaveNet for time series

from __future__ import print_function, division
from keras.layers import Conv1D, Input, Add, Activation, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from keras import optimizers


def Casual_CNN_Kernel(filter_cnt, filter_length, dilation, l2_flag):
    
    # define individual kernel structure
    # model is defined using function API of Keras
    
    def CNN_Kernel_ind(input_):
        
        residual =    input_
        
        output =   Conv1D(filters = filter_cnt, 
                            kernel_size = filter_length, 
                            dilation_rate = dilation,  
                            padding = 'causal', # do not include any information happen in the future
                            activation = 'linear',
                            use_bias=False,
                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.03, seed=11), 
                            kernel_regularizer=l2(l2_flag))(input_)  
        # adding activation layer                 
        output =   Activation('selu')(output)
        
        # skipped time points
        output2 =    Conv1D(1,1, 
                             activation='linear', 
                             use_bias=False, 
                             kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=11), 
                             kernel_regularizer=l2(l2_flag))(output)
        # time points used in conv
        output_temp =  Conv1D(1,1, 
                             activation='linear', 
                             use_bias=False, 
                             kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=11), 
                             kernel_regularizer=l2(l2_flag))(output)
                      
        output1 = Add()([residual, output_temp])
        
        return output1, output2
    
    return CNN_Kernel_ind


def Casual_CNN_Model(length, LR_S):
    
    input = Input(shape=(length,1))
    #    Casual_CNN_Kernel(filter_cnt, filter_length, dilation, l2_flag):
    #  6 layers structure. can be modified
    layer_1a, layer_1b = Casual_CNN_Kernel(32,2,1,0.001)(input)    
    layer_2a, layer_2b = Casual_CNN_Kernel(32,2,2,0.001)(layer_1a) 
    layer_3a, layer_3b = Casual_CNN_Kernel(32,2,4,0.001)(layer_2a)
    layer_4a, layer_4b = Casual_CNN_Kernel(32,2,8,0.001)(layer_3a)
    layer_5a, layer_5b = Casual_CNN_Kernel(32,2,16,0.001)(layer_4a)
    layer_6a, layer_6b = Casual_CNN_Kernel(32,2,32,0.001)(layer_5a)
    layer_6b = Dropout(0.8)(layer_6b) 
#    layer_7a, layer_7b = Casual_CNN_Kernel(32,2,64,0.001)(layer_6a)
#    layer_7b = Dropout(0.8)(layer_7b) 

#    l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])
    layer_8 =   Add()([layer_1b, layer_2b, layer_3b, layer_4b, layer_5b, layer_6b])
    
    layer_9 =   Activation('relu')(layer_8)
           
    layer_21 =  Conv1D(1,1, 
                       activation='linear', 
                       use_bias=False, 
                       kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=11), 
                       kernel_regularizer=l2(0.001))(layer_9)

    model = Model(input=input, output=layer_21)    
    adam = optimizers.Adam(lr=LR_S)
    model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    
    return model
