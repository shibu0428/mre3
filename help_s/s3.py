import numpy as np
import matplotlib.pyplot as plt
np.random.seed(59)

a = 0.1
b = 1
x0 = 10
T = 0.8
trials = 1000

# 試験関数
def phi(x):
    return x


def true_solution(t, W):
    return x0*np.exp((a-0.5*b**2)*t+b*W)

alpha = a - 0.5*(b**2)

# Euler–Maruyama 法 (Y空間)
def euler_Y(y0, alpha, b, dt, W):
    n_steps = len(W)
    Y = np.zeros(n_steps+1)
    Y[0] =y0 
    for k in range(n_steps):
        Y[k+1] = Y[k] + alpha*dt + b*W[k]
    return Y

#標準化
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/(xstd)
    return zscore


# 逆変換: X = exp(Y)
def Y_to_X(Y):
    return np.exp(Y)

def fin_exact_solution(trials,T,n):
    dt=T/n
    print(f'{dt:.60f}')
    t_values=np.linspace(0, T, n+1)
    exact_solutions=np.zeros(trials)
    for i in range(trials):
        #W_zoubun=np.random.randn(n)*np.sqrt(dt)
        W_zoubun=zscore(np.random.randn(n))*np.sqrt(dt)
       
        W_sum = np.cumsum(W_zoubun)
        W_full = np.concatenate(([0], W_sum))
        exact_solutions[i]=(phi(true_solution(t_values, W_full)[-1]))
    #print(n,np.mean(W_zoubun))
    mean_exact_solution = np.mean(exact_solutions)
    return mean_exact_solution

n_values=[200,500,1000,4000,8000,10000,20000]                               #200, 400, 600, 800, 1000,1500,3000,6000,10000
#n_values = np.logspace(np.log10(10), np.log10(100000), num=20, dtype=int)
#n_values = sorted(set(n_values))  # 重複を避けてソート
#n_values=range(10,100,50)
mean_fins_euler_Y = []


mean_exact_solution=fin_exact_solution(1000,T,100)



# シミュレーションと誤差計算
for n in n_values:
    #print(n,"start")
    dt=T/n
    t_values=np.linspace(0, T, n+1)
    fin_euler_Y = []
    fin_euler_Ys=[]
    print(f'{1.0:.60f}')

    for trial in range(trials):
        #W_zoubun=np.random.randn(n)*np.sqrt(dt)
        W_zoubun=zscore(np.random.randn(n))*np.sqrt(dt)
        #print(f'{W_zoubun[0]:.60f}')
        #Yオイラー法
        Y_euler = euler_Y(np.log(x0), alpha, b, dt, W_zoubun)
        #fin_euler_Ys.append(Y_euler[n])
        fin_euler_Y.append(phi(Y_to_X(Y_euler[n])))


    #mean_fin_euler_Ys=np.mean(fin_euler_Ys)
    mean_fin_euler_Y = np.mean(fin_euler_Y)
    #print(Y_euler[n],mean_fin_euler_Y,mean_exact_solution)
    mean_fins_euler_Y.append(np.abs(mean_fin_euler_Y-mean_exact_solution))
    #print(np.sign(mean_fin_euler_Y-mean_exact_solution))
    #mean_fins_euler_Y.append(np.abs(phi(Y_to_X(mean_fin_euler_Ys))-mean_exact_solution))
    #mean_fins_euler_Y.append(np.abs(mean_fin_euler_Y-phi(exact_solution[-1])))
    #mean_fins_euler_Y.append(np.abs(np.log(mean_fin_euler_Y)-np.log(mean_fin_exact_solution)))
#print("mean_fin_exs",mean_fin_exact_solution)
plt.figure(figsize=(8, 6))
plt.loglog(n_values, mean_fins_euler_Y, 'o-', label='new', color='red')
# 理論的な収束率の調整 

scaling_factor_euler_Y = mean_fins_euler_Y[0] / (n_values[0]**(1.0))

n_array = np.array(n_values)
print(mean_fins_euler_Y)
plt.loglog(n_array, scaling_factor_euler_Y * n_array**(1.0), 'b--', label='-1 (Euler_Y)')

plt.xlabel('Number of time steps')
plt.ylabel('Average Error')
plt.legend()


plt.grid(False)
plt.axvline(x=1000)
plt.axvline(x=5000)
plt.axvline(x=10000)
plt.show()