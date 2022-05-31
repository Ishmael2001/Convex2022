import numpy as np
import math
import matplotlib.pyplot as plt
from time import time


class LBFGS: 
    def __init__(self,x0,err,alpha,beta,m):
        self.x0=x0
        self.err=err
        self.alpha=alpha
        self.beta=beta
        self.f_iter=[]
        self.iters=0
        self.iter=[]
        self.time=[]
        self.m=m
        self.first=True
        self.memory_s=[]
        self.memory_y=[]
        self.Hk=np.mat(np.identity(2000))
        
    def f(self,x):
        ans=0  #a=1
        for i in range(1,1001):
            ans+=((x[2*i-1][0]-x[2*i-2][0]**2)**2+(1-x[2*i-2][0])**2)[0,0]
        return ans

    def g(self,x):
        gradient=[]
        for i in range(1,1001):
            gradient.append((4*x[2*i-2][0]**3-4*x[2*i-2][0]*x[2*i-1][0]+2*x[2*i-2][0]-2)[0,0])
            gradient.append((2*x[2*i-1][0]-2*x[2*i-2][0]**2)[0,0])
        return np.mat(gradient).transpose()
        
    def step(self):       
        fk=self.f(self.x0)
        gk=self.g(self.x0)
        if (gk.transpose()*gk)[0,0]<self.err**2:
            return True
        self.f_iter.append(fk)
        self.time.append(time()-self.t0)
        self.iter.append(self.iters)
        self.iters+=1

        #确定H0
        if self.first==True:
            Hk=self.Hk
            self.first=False
        else:
            gama=(self.memory_s[-1].transpose()*self.memory_y[-1])/(self.memory_y[-1].transpose()*self.memory_y[-1])
            Hk=np.mat(np.diag([gama[0,0]]*2000))
        
        #两步循环计算pk
        q=self.g(self.x0)
        a=[]
        for i in range(min(self.m,len(self.memory_s))):
            ai=self.memory_s[-1-i].transpose()*q/(self.memory_y[-1-i].transpose()*self.memory_s[-1-i])
            a.append(ai)
            q=q-ai[0,0]*self.memory_y[-1-i]
        p=Hk*q
        for i in range(min(self.m,len(self.memory_s))):
            bi=self.memory_y[i].transpose()*p/(self.memory_y[i].transpose()*self.memory_s[i])
            p=p+(a[-1-i]-bi)[0,0]*self.memory_s[i]
        
        #满足wolfe条件:每次递减0.02
        alpha=1
        flag=False
        while flag==False:
            if self.f(self.x0-alpha*p)<=self.f(self.x0)-0.25*alpha*self.g(self.x0).transpose()*p:
                if -self.g(self.x0-alpha*p).transpose()*p>=-0.75*self.g(self.x0).transpose()*p:
                    flag=True
            alpha-=0.02

        x_next=self.x0-alpha*p
        self.memory_y.append(self.g(x_next)-self.g(self.x0))
        self.memory_s.append(x_next-self.x0)
        if len(self.memory_s)>=self.m: 
            self.memory_s=self.memory_s[1:]
            self.memory_y=self.memory_y[1:]
        self.x0=x_next
        #print((self.g(self.x0).transpose()*self.g(self.x0))[0,0])
        if (self.g(self.x0).transpose()*self.g(self.x0))[0,0]<self.err**2:
            return True
        else:
            return False
    
    def opt(self):
        self.t0=time()
        while True:
            if self.step():
                break
        print("选取内存m={}".format(self.m))
        print("迭代次数={}".format(self.iters))
        print("程序用时={}".format(self.time[-1]))
        print("--------------------")
    
    def plot_fi(self):
        plt.title("L-BFGS algorithm")
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.plot(self.iter,self.f_iter)
        plt.show()

    def plot_fi_t(self):
        plt.title("L-BFGS algorithm")
        plt.xlabel('time')
        plt.ylabel('f(x)')
        plt.plot(self.time,self.f_iter)
        plt.show()

err=1e-5
alpha=0.9
beta=0.5
m=3
for i in range(3,21):
    x0=np.mat([-1 for i in range(2000)]).transpose()
    lbfgs1=LBFGS(x0,err,alpha,beta,i)
    lbfgs1.opt()
