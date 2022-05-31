import numpy as np
import math
import matplotlib.pyplot as plt

class CGA: #conjugate gradient method
    def __init__(self,x0,err,alpha,beta,method):
        self.x0=x0
        self.err=err
        self.alpha=alpha
        self.beta=beta
        self.log_loss=[]
        self.iters=0
        self.iter=[]
        self.buchang=[]
        self.method=method
        self.dk=0
        
    def f(self,x):  #选取a=1
        ans=0
        for i in range(1,51):
            ans+=(x[2*i-1]-x[2*i-2]**2)**2+(1-x[2*i-2])**2
        return ans

    def g(self,x):
        gradient=[]
        for i in range(1,51):
            gradient.append(4*x[2*i-2]**3-4*x[2*i-2]*x[2*i-1]+2*x[2*i-2]-2)
            gradient.append(2*x[2*i-1]-2*x[2*i-2]**2)
        return np.array(gradient)
    
    def step(self):       
        fk=self.f(self.x0)
        gk=self.g(self.x0)
        gk2=np.dot(gk,gk)
        if gk2<self.err**2:
            return True

        self.log_loss.append(math.log(fk))
        self.iter.append(self.iters)
        self.iters+=1
        t=1
        while self.f(self.x0+t*self.dk)>self.f(self.x0)+self.alpha*t*np.dot(gk,self.dk) and t>0.00000000000001:
            t*=self.beta
        self.x0=self.x0+t*self.dk
        self.buchang.append(t)
        gk_=self.g(self.x0)
        if np.dot(gk_,gk_)<self.err**2:
            return True

        if self.method=="HS":
            bk=(gk_ @ (gk_-gk))/(self.dk @ (gk_-gk))
        elif self.method=="PR":
            bk=gk_@(gk_-gk)/(gk_@gk)
        elif self.method=="FR":
            bk=gk_@gk_/(gk@gk)
        self.dk=-1*gk_+bk*self.dk
        return False
    
    def opt(self):
        gk=self.g(self.x0)
        self.dk=-1*gk
        while True:
            if self.step():
                break
    
    def plot_fi(self):
        if self.method=="HS":
            plt.title("CGA:Hestenes-Stiefel formula")
        elif self.method=="PR":
            plt.title("CGA:Polak-Ribiere formula")
        elif self.method=="FR":
            plt.title("CGA:Fletcher-Reeves formula")
        plt.xlabel('iterations')
        plt.ylabel('log(f(x)-p*)')
        plt.plot(self.iter,self.log_loss)
        plt.show()

    def plot_ft(self):
        if self.method=="HS":
            plt.title("CGA:Hestenes-Stiefel formula")
        elif self.method=="PR":
            plt.title("CGA:Polak-Ribiere formula")
        elif self.method=="FR":
            plt.title("CGA:Fletcher-Reeves formula")
        plt.xlabel('iterations')
        plt.ylabel('a')
        plt.plot(self.iter,self.buchang)
        plt.show()

x0=np.array([-1 for i in range(100)])
err=0.000005
alpha=0.5
beta=0.5
method=["HS","PR","FR"]
for i in range(3):
    cga1=CGA(x0,err,alpha,beta,method[i])
    cga1.opt()
    cga1.plot_fi()
    cga1.plot_ft()