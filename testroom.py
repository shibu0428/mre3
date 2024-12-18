import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

a=np.zeros((3,7))
num=0
for i in range(3):
    for j in range(7):
        a[i][j]=num
        num+=1

print(a)
print()


a=np.roll(a,-7)

print(a)