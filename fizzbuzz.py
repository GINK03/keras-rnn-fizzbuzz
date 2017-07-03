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


inputs      = Input(shape=(10, 11))
encoded     = GRU(256)(inputs)
encoded     = Dense(512, activation='relu')( encoded )
encoded     = BN()( encoded )
encoder     = Model( inputs, encoded )
x           = RepeatVector(50)( encoded )
x           = Bi(GRU(256, return_sequences=True))(x)
x           = BN()(x)
x           = TD(Dense(256, activation='relu'))(x)
decoded     = Dense(3, activation='sigmoid')( Flatten()(x) )

fizzbuzz     = Model(inputs, decoded)
fizzbuzz.compile(optimizer=Adam(), loss='binary_crossentropy')

buff = None
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
batch_callback = LambdaCallback(on_batch_end=lambda batch,logs: callback(batch,logs) )

def train():

  optims = [Adam(), SGD(), RMSprop()]

  count = 0
  for i in range(50):
    for name in sorted(glob.glob("dataset/*.pkl")):
      datanum = re.search(r'(\d+)', name).group(1)
      if int(datanum) % 5 == 0: 
        continue
      count += 1
      data_pair = pickle.loads( open( name, "rb" ).read() )
      Xs, Ys    = data_pair
      target_optim = random.choice(optims)
      print("👻train dataset No.", name, "optimizer", target_optim)
      fizzbuzz.optimizer = target_optim
      fizzbuzz.fit(Xs, Ys, callbacks=[batch_callback])
      fizzbuzz.save("models/%09d_%09.5f.h5"%(count, buff['loss']))


if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
