
import msgpack
import random
import pickle
import glob
import sys

class OR(object):
  fizz     = [1.0, 0.0, 0.0]
  buzz     = [0.0, 1.0, 0.0]
  fizzbuzz = [1.0, 1.0, 0.0]
  path     = [0.0, 0.0, 1.0]
  def __init__(self):
    ...

def fizzbuzz():
  ice   = [i for i in range(100000)]
  random.shuffle(ice)

  ice_pack = {}
  for e, i in enumerate(ice):
    e = e//1024
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

      pair = (inputs, output)
      print(pair)
      pairs.append( pair ) 
    open('dataset/dataset_%09d.pkl', 'wb').write( pickle.dumps(pairs) )

  

def build_dict():
  char_freq = {}
  for eg, name in enumerate( glob.glob("dataset/*.pkl") ):
    if eg%500 == 0:
      print( eg, name )
    mms, raws = pickle.loads( open(name, "rb").read() )
    for char in list( "".join( [mms, raws] ) ):
      if char_freq.get( char ) is None:
        char_freq[char] = len( char_freq )
  open("char_freq.pkl", "wb").write( pickle.dumps(char_freq) )

  char_index = {}
  for char, freq in char_freq.items():
    char_index[char] = len( char_index )
  open("char_index.pkl", "wb").write( pickle.dumps(char_index) )
if __name__ == '__main__':
  if '--step1' in sys.argv:
    fizzbuzz()

  if '--step2' in sys.argv:
    build_dict()
 
