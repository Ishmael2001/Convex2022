import numpy as np
import math
import matplotlib.pyplot as plt

def f(a,x0):  #求函数值
    value=0
    for i in a:
        value+=(math.exp(np.dot(i,x0))+math.exp(-np.dot(i,x0)))
    return value

def gradient(a,x0):  #求梯度
    g=[]
    for i in range(len(x0)):
        value=0
        for j in range(len(x0)):
            value+=a[j][i]*math.exp(np.dot(a[j],x0))-a[j][i]*math.exp(-np.dot(a[j],x0))
        g.append(value)
    return np.array(g)

def Backtracking(alpha,beta,_a,_x):
    #初始设定
    a=_a
    x=_x
    t=1
    p=f(a,np.array([0 for i in range(10)]))
    reverse_search_direction = gradient(a,x)
    min_err=0.005  #迭代结束时的最小误差
    time=0  #从第0次迭代开始

    #在过程中收集以下参数
    objvalue=[(f(a,x)-p)**2]  #迭代过程中的目标函数值
    steplength=[t]  #迭代过程中的步长
    grad=[np.linalg.norm(reverse_search_direction)]  #迭代过程中的梯度二范数
    
    while np.linalg.norm(reverse_search_direction)>min_err:
        time+=1
        t=1
        while f(a,x-t*reverse_search_direction)>f(a,x)-alpha*t*np.dot(reverse_search_direction,reverse_search_direction):
            t=beta*t
        x=x-t*reverse_search_direction
        reverse_search_direction = gradient(a,x)
        objvalue.append((f(a,x)-p)**2)
        steplength.append(t)
        grad.append(np.linalg.norm(reverse_search_direction))
    
    #return(time)

    times=[i for i in range(0,time+1)]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    plt.title('Backtracking算法:alpha={},beta={}'.format(alpha,beta))  # 折线图标题
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数:(f(x)-p*)**2')
    plt.plot(times,objvalue)
    plt.show()
    plt.title('Backtracking算法:alpha={},beta={}'.format(alpha,beta))  # 折线图标题
    plt.xlabel('迭代次数')
    plt.ylabel('步长')
    plt.plot(times,steplength)
    plt.show()
    plt.title('Backtracking算法:alpha={},beta={}'.format(alpha,beta))  # 折线图标题
    plt.xlabel('迭代次数')
    plt.ylabel('梯度二范数')
    plt.plot(times,grad)
    plt.show()

#不妨取m=10,a和x维度均为10,a各分量服从N(0,0.1^2),x初始各分量取1
a=[]
for i in range(10):
    a.append(0.1*np.random.randn(10))
a=np.array(a)
x=np.ones(10)

Backtracking(0.9,0.5,a,x)
'''for i in [0.8,0.85,0.9,0.95]:
    for j in [0.8,0.85,0.9,0.95]:
        iter=Backtracking(i,j,1,a,x)
        print("alpha={},beta={},t=1下需要迭代{}次".format(i,j,iter))
        '''