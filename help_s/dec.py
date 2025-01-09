from decimal import Decimal, getcontext
import numpy as np

n=10
r=np.random.randn(n)
getcontext().prec = 64
r_dec=[]

sum=Decimal('0')
std=Decimal('0')
for j in range(len(r)):
    r_dec.append(Decimal(str(r[j])))
    sum+=r_dec[j]

heikin=Decimal(str(sum/Decimal(str(n))))
for k in range(len(r)):
    std+=(r_dec[k]-heikin)*(r_dec[k]-heikin)
s=Decimal(std/Decimal(str(n)))

r_nolm=[]
for l in range(len(r)):
    r_nolm.append((r_dec[l]-heikin)/(s))

print(r_nolm)