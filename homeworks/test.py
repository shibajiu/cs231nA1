import numpy as np
import operator

d = np.arange(3).reshape(3,-1)
e=np.arange(3).reshape(-1,3)
f=d*e
print(f)
print(f.sum(axis=0))
a=[0,1,2]
print(f[:,1])
