import numpy as np

motion_len=15
b=np.zeros([motion_len,motion_len])
b[2][3]=1
print(b)