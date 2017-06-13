import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    P = len(series) # P is length of series
    T = window_size # window_size. Have renamed variables to match the notations in explanation above
    # number of pairs. Actual size of input with last input char removed is P-1. For a series of P-1 chars and a 
    # window size of T, we will get (P-1)-T+1 = P-T pairs
    no_pairs = P-T 
    
    X = []
    y = []
    for i in range(no_pairs):
        # slice data for ith pair and append
        X.append(np.array(series[i:i+T]))
        y.append(series[i+T])
    
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y    
    
# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential() # create a sequential model
    model.add(LSTM(5, input_shape=(window_size, step_size)))  # add LSTM with 5 hidden units and required input_type
    model.add(Dense(1)) # we add Dense with output of 1 and no activation as we need regression output

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique_chars = set(text) # get unqiue chars
    #print(unique_chars)

    # remove as many non-english characters and character sequences as you can
    # list of chars to remove
    non_eng_chars = ['-', '(', ')', '8', '%', "'", '2', '9', 'è', '@', '$', '"', 'â', '4', 
                    '3', 'é', '0', '&', '1', '6', '/', '5', '7', '*', 'à']
    # go over non english characters and remove each one from input text storing result back in `text`
    for char in non_eng_chars:
        text = text.replace(char, ' ')


        
    # shorten any extra dead space created above
    text = text.replace('  ',' ')    
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    P = len(text) #length of text
    T = window_size #Size of window
    M = step_size #Size of step
    # no of pairs
    # same formula as that of 'VALID' padding in COnvolution
    # refer https://www.tensorflow.org/api_guides/python/nn#Convolution
    P_inp = P-1 # as the last symbol in text input text[P] is the output of final pair. So input text length is 'len(text)-1'
    num_pairs = np.ceil(float(P_inp - T + 1) / float(M))
    num_pairs = int(num_pairs)
    for i in range(num_pairs):
        inp = text[i*M:i*M+T] #i is the step number. so for ith step we take T character starting at (i*M) 
        out = text[i*M+T]
        inputs.append(inp)
        outputs.append(out)
    
    return inputs,outputs
