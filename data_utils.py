
import msgpack
import random
import pickle
import glob
import sys
import numpy as np
index_char = {}
char_index = {}
for i,char in enumerate(list('0123456789 ')):
  index_char[i]    = char
  char_index[char] = i

class OR(object):
  fizz     = [1.0, 0.0, 0.0]
  buzz     = [0.0, 1.0, 0.0]
  fizzbuzz = [1.0, 1.0, 0.0]
  path     = [0.0, 0.0, 1.0]
  def __init__(self):
    ...

def fizzbuzz():
  ice   = [i for i in range(100000*5)]
  random.shuffle(ice)

  ice_pack = {}
  for e, i in enumerate(ice):
    e = e//(1024*5)
    if ice_pack.get(e) is None:
      ice_pack[e] = []
    ice_pack[e].append( i )

  for ice, pack in ice_pack.items():
    pairs = []
    for i in pack:
      inputs = "% 10d"%i
      output = None
      if i%15 == 0:
        output = OR.fizzbuzz
      elif i%5 == 0:
        output = OR.buzz
      elif i%3 == 0:
        output = OR.fizz
      else:
        output = OR.path
      
      # inputs:string -> inputs:tensor 
      input_tensor = []
      for char in list(inputs):
        onehot = [0.0]*11
        onehot[char_index[char]] = 1.0
        input_tensor.append( onehot )
      input_tensor  = np.array( list(reversed(input_tensor) ) )
      
      # output:vector -> output:nparray
      output_tensor = np.array( output )
      pair = (input_tensor, output_tensor)
      # print(pair)
      pairs.append( pair ) 
    Xs = np.array( [xs for xs, ys in pairs] )
    Ys = np.array( [ys for xs, ys in pairs] )
    data_pair = (Xs, Ys) 
    open('dataset/dataset_%09d.pkl'%ice, 'wb').write( pickle.dumps(data_pair) )

if __name__ == '__main__':
  if '--step1' in sys.argv:
    fizzbuzz()
 
