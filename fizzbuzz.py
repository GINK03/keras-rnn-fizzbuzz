from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.models          import Model
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re


timesteps   = 200
inputs      = Input(shape=(timesteps, 100))
encoded     = GRU(512)(inputs)
inputs_a    = inputs
inputs_a    = Dense(2048)(inputs_a)
inputs_a    = BN()(inputs_a)
encoder     = Model(inputs, inputs_a)

x           = RepeatVector(timesteps)( inputs_a )
x           = Bi(LSTM(512, return_sequences=True))(x)
decoded     = TD(Dense(100, activation='softmax'))(x)

fizzbuzz     = Model(inputs, decoded)
fizzbuzz.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  c_i = pickle.loads( open("char_index.pkl", "rb").read() )
  xss = []
  yss = []
  for eg, name in enumerate(glob.glob("dataset/*.pkl")):
    mms, raw = pickle.loads( open( name, "rb" ).read() )
    sss = eval( raw )
    print( sss["txt"] )
    print( mms, raw )
   
    x = [" "]*200
    for i, m in enumerate(mms):
      x[i] = m
    xs = [ [0.0]*len(c_i) for i in range(200) ]
    for i in range(len(x)):
      xs[i][ c_i[x[i]] ] = 1.0
    xs = list(reversed( xs ))

    y = [" "]*200
    for i, m in enumerate(raw):
      y[i] = m
    ys = [ [0.0]*len(c_i) for i in range(200) ]
    for i in range(len(x)):
      ys[i][ c_i[y[i]] ] = 1.0
   
    xss.append( xs )
    yss.append( ys )
  
  xss = np.array( xss )
  yss = np.array( yss )

  for i in range(10000):
    parser.fit(xss, yss)
    parser.save("models/%9f_%09d.h5"%(buff['loss'], i))
if __name__ == '__main__':
  if '--train' in sys.argv:
    train() 
