import matplotlib.pyplot as plt
import numpy as np

def true_solution(t, W):

    n = len(t) - 1  # ステップ数
    dt = t[1] - t[0]

    drift = 0.25 * t**2  # shape (n+1,)

  
    diffuse = np.zeros(n+1)
    for k in range(n):
        t_k = t[k]
        dWk = W[k+1] - W[k]
        diffuse[k+1] = diffuse[k] + np.sqrt(t_k)*dWk

    X = x0 * np.exp(drift + diffuse)
    return X


def alpha_func(t):
    return 0.5 * t

def euler_Y(y0, dt, W_incr):
    n = len(W_incr)
    Y = np.zeros(n+1)
    Y[0] = y0
    for k in range(n):
        t_k = k * dt
        Y[k+1] = Y[k] + alpha_func(t_k)*dt + b(t_k)*W_incr[k]
    return Y

def Y_to_X(Y):
    return np.exp(Y)

def a(t):
    return 0.5*t

def b(t):
    return np.sqrt(t)

def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    z=(x - xmean)/xstd
    return z

#np.random.seed(40228)  # 乱数のシードを設定

plt.figure(figsize=(8, 6))
T=2.5
x0=1.0
N=10000
dt=T/N

tryals=10000
t = np.linspace(0, T, N+1)

Xts=[]

print("true solution")
for i in range(2500):
    if i%100==0:
        print(i)
    dW =np.random.randn(N)*np.sqrt(dt)
    W_full = np.concatenate(([0.0], np.cumsum(dW)))
    Xt = true_solution(t, W_full)
    Xts.append(Xt[-1])
    #plt.plot(t, Xt, label='True solution')
'''
dW=zscore(dW)*np.sqrt(dt)
Y  = euler_Y(np.log(x0), dt, dW)
Xe = Y_to_X(Y)
plt.plot(t, Xe  , label='Euler zs true')
'''
'''
Y  = euler_Y(np.log(x0), dt, dW)
Xe = Y_to_X(Y)
plt.plot(t, Xe, label='Euler')
'''
ezs=[]
for i in range(2500):
    if i%100==0:
        print(i)
        #plt.plot(t, Xe  , label='Euler zscore'+str(i/500))
    dW =np.random.randn(N)*np.sqrt(dt)
    dW=zscore(dW)*np.sqrt(dt)
    Y  = euler_Y(np.log(x0), dt, dW)
    Xe = Y_to_X(Y)
    
    ezs.append(Xe[-1])

print("true solution: ",np.mean(Xts))
print("eular_Y zscore mean: ",np.mean(ezs))
#plt.legend()   
#plt.show()
''' 
plt.plot(t, Xe, label='Euler zscore')
plt.plot(t, Xt, label='True solution')
plt.legend()
plt.show()
'''