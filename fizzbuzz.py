from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.models          import Model
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply, concatenate, dot
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.merge    import Concatenate, Dot
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import json
import glob
import copy
import os
import re
import time

inputs       = Input(shape=(10, 11))
encoded1     = Bi( GRU(256, activation='relu') )(inputs)
encoded1     = Dense(512, activation='relu')( encoded1 )
encoded1_1x  = Reshape((1,512,))(encoded1)

#encoded2     = Dense(512, activation='relu')( Flatten()( inputs ) )
#encoded2     = Dense(512, activation='relu')( encoded2 )
#encoded2_1x  = Reshape((1,512,))(encoded2)

#mult         = multiply( [encoded1, encoded2] ) # ここを加えると、性能の劣化が見られる
#conc         = concatenate( [encoded1_1x, encoded2_1x] )
decoded      = Dense(3, activation='sigmoid')( Flatten()(encoded1_1x) )

fizzbuzz     = Model(inputs, decoded)
fizzbuzz.compile(optimizer=Adam(), loss='binary_crossentropy')

buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
  # logging for all epoch loss
  with open('loss_%s.log'%now, 'a+') as f:
    f.write('%s\n'%str(buff))
   
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )

class CURRICULUM:
  EPOCH = [50, 30, 20, 10, 5, 1]
  EPOCH = [1]

  @staticmethod
  def GET():
    if len(CURRICULUM.EPOCH) > 0:
      return CURRICULUM.EPOCH.pop(0)
    else:
      return 1
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
      fizzbuzz.fit(Xs, Ys, epochs=CURRICULUM.GET(), callbacks=[batch_callback])
      fizzbuzz.save("models/%09d_%09.5f.h5"%(count, buff['loss']))

class RURE:
  @staticmethod
  def CH(vec, inputs):
    res = ""
    if vec[0] > 0.5 and vec[1] > 0.5:
      res = "Fizz Buzz"
    elif vec[0] > 0.5:
      res = "Fizz"
    elif vec[1] > 0.5:
      res = "Buzz"
    else:
      res = "Path"
    cl = list("0123456789 ")
    orig = ""
    for inp in inputs:
      ii = max( [(i, f) for i, f in enumerate(inp)], key=lambda x:x[1])
      i = ii[0]
      orig += cl[i]  
    orig = orig[::-1]
    return (res, orig)
def predict():
  model = sorted(glob.glob('models/*')).pop()
  fizzbuzz.load_weights(model)
  for name in sorted(glob.glob("dataset/*.pkl")):
    datanum = re.search(r'(\d+)', name).group(1)
    if int(datanum) % 5 != 0: 
      continue
    data_pair = pickle.loads( open( name, "rb" ).read() )
    Xs, Ys    = data_pair
    for (real, pred, inputs) in zip(Ys.tolist(), fizzbuzz.predict(Xs).tolist(), Xs.tolist()):
      origR, orig = RURE.CH(real, inputs) 
      predR, orig = RURE.CH(pred, inputs)
      print( orig, "original", origR, "predict", predR, origR == predR)
  

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
  if '--predict' in sys.argv:
    predict()
