import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.optimize import minimize


class CO:  #Constrained Optimization
    def __init__(self,A,b,x,alpha,beta,err,judge=False,lam=np.zeros(100),be=0.00001):
        self.A=A
        self.b=b
        self.alpha=alpha
        self.beta=beta
        self.err=err  #衡量结果的误差
        self.steps=0  #迭代的步数
        self.everystep_f=[]
        self.P=np.eye(500) -A.transpose() @ np.linalg.inv(A @ A.transpose()) @ A
        self.x = self.P @ x + A.transpose() @ (np.linalg.inv(A @ A.transpose()) @ b)  #满足Ax=b的条件的特值
        self.sol = self.x.copy()  #一个满足初始条件的特解
        u, s, vt = np.linalg.svd(self.A)
        self.zero_space = vt[100:].transpose()
        self.judge=judge
        self.lam=lam
        self.be=be
        
    def f(self,x):
        return np.log(np.sum(np.exp(x)))
    def g(self,x):
        return np.exp(x) / np.sum(np.exp(x))
    def H(self,x):
        g=np.exp(x) / np.sum(np.exp(x))
        return np.diag(g) - np.outer(g, g)
    def lsearch(self, d, f, g, func):
        t = 1
        while func(self.x+t*d) > f+self.alpha*t*np.dot(d, g):
            t *= self.beta
        return t
    def fe(self, z):
        return self.f(self.sol + self.zero_space @ z)
    def ge(self, z):
        return self.zero_space.transpose() @ self.g(self.sol + self.zero_space @ z)
    def l(self,x):
        return self.f(x) + np.dot(self.lam, self.A@self.x-self.b) + self.be/2*(np.linalg.norm(self.A@self.x-self.b, 2)**2)
    def gl(self,x):
        return self.g(x) + self.A.T@self.lam + self.be*self.A.T@(self.A@self.x-self.b)
    def step(self,method):
        if method=="DPG": #Direct Projected Gradient
            f=self.f(self.x)
            g=self.g(self.x)
            d=-self.P@g        
        elif method=="EEC":  #Eliminating Equality Constraints
            if self.steps==0:
                self.x = np.zeros(400)
            f = self.fe(self.x)
            g = self.ge(self.x)
            d = -g
        elif method=="NM":  #Newton’s Method with Equality Constraint
            f = self.f(self.x)
            g = self.g(self.x)
            H = self.H(self.x)
            KKT_mat1 = np.concatenate([H, self.A.T], axis=1)
            KKT_mat2 = np.concatenate([self.A, np.zeros((100, 100))], axis=1)
            KKT_mat = np.concatenate([KKT_mat1, KKT_mat2], axis=0)
            g0 = np.concatenate([-g, np.zeros(100)])
            d = np.linalg.solve(KKT_mat, g0)[:500]
        elif method=="DA":  #Dual Ascent
            f = self.f(self.x)
            L = self.l(self.x) 
            g = self.gl(self.x)
            d = -g
            l = d
        print(f)
        self.everystep_f.append(f)
        if np.dot(d,d)<self.err**2:
            return False
        if method=="DA":  #Dual Ascent
            if self.judge==True:
                self.x=minimize(self.l,self.x,jac=self.gl).x
            else:
                alpha=self.lsearch(d,L,g,self.l)
                self.x+=alpha*d
            self.lam+=self.be*(self.A@self.x-self.b)    
        else:
            alpha = self.lsearch(d,f,g,self.f)
            self.x += alpha*d
        return True
    def opt(self,method):
        flag = True
        self.steps=0
        while flag:
            flag = self.step(method)
            self.steps += 1
            if self.steps==10:
                break
        self.pstar = self.everystep_f[-1]
    
A = np.random.normal(0, 1, (100, 500))
b = np.random.normal(0, 1, 100)
x = np.random.normal(0, 0.1, 500)


trial3=CO(A,b,x,0.9,0.5,0.001)
trial3.opt("NM")
c=trial3.everystep_f

trial4=CO(A,b,x,0.9,0.5,0.001)
trial4.opt("DA")
d=trial4.everystep_f

trial1=CO(A,b,x,0.9,0.5,0.001)
trial1.opt("DPG")
a=trial1.everystep_f

trial2=CO(A,b,x,0.9,0.5,0.001)
trial2.opt("EEC")
b=trial2.everystep_f

plt.plot(a)
plt.plot(b)
plt.plot(d)
plt.xlabel('iterations')  # x轴标题
plt.ylabel('f')
plt.legend(['DPG', 'EEC','DA'],loc="best")  # 设置折线名称
plt.show()  # 显示折线图
plt.plot(c)
plt.xlabel('iterations')  # x轴标题
plt.ylabel('f')
plt.legend("NM",loc="best")  # 设置折线名称
plt.show()  # 显示折线图
