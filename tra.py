
import re
import sys

for e, line in enumerate(sys.stdin):
  line = line.strip()
  loss = re.search(r'(\d.\d{1,})', line).group(1)
  print( e, loss ) 
