import numpy as np
import math
import matplotlib.pyplot as plt
from time import time

class DNM: #Damped Newton's Method
    def __init__(self,x0,err,alpha,beta):
        self.x0=x0
        self.err=err
        self.alpha=alpha
        self.beta=beta
        self.log_loss=[]
        self.t=[]
        self.iters=0
        self.iter=[]
        
    def f(self,x):
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

    def g(self,x):
        gradient=[400*x[0]**3-400*x[0]*x[1]+2*x[0]-2,200*x[1]-200*x[0]**2]
        return np.array(gradient)

    def H(self,x):
        hessian = np.zeros((2,2))
        hessian[0][0]=1200*x[0]**2-400*x[1]+2
        hessian[0][1]=-400*x[0]
        hessian[1][0]=-400*x[0]
        hessian[1][1]=200
        return hessian
    
    def step(self):
        fx=self.f(self.x0)
        gk=self.g(self.x0)
        Hk=self.H(self.x0)
        self.log_loss.append(math.log(fx))
        self.t.append(time()-self.t0)
        self.iter.append(self.iters)
        self.iters+=1

        gk2=np.dot(gk,gk)
        if gk2<self.err**2:
            return True
        dk=-np.linalg.inv(Hk) @ gk
        t=1
        while self.f(self.x0+t*dk)>self.f(self.x0)+self.alpha*t*np.dot(gk,dk):
            t*=self.beta
        self.x0=self.x0+t*dk
        return False
    
    def opt(self):
        self.t0=time()
        while True:
            if self.step():
                break
    
    def plot_fi(self):
        plt.title("Damped Newton's Method")
        plt.xlabel('iterations')
        plt.ylabel('log(f(x)-p*)')
        plt.plot(self.iter,self.log_loss)
        plt.show()
    
    def plot_ft(self):
        plt.title("Damped Newton's Method")
        plt.xlabel('time')
        plt.ylabel('log(f(x)-p*)')
        plt.plot(self.t,self.log_loss)
        plt.show()

class GNM: #Gauss Newton's Method
    def __init__(self,x0,err):
        self.x0=x0
        self.err=err
        self.log_loss=[]
        self.t=[]
        self.iters=0
        self.iter=[]
        
    def f(self,x):
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

    def g(self,x):
        gradient=[400*x[0]**3-400*x[0]*x[1]+2*x[0]-2,200*x[1]-200*x[0]**2]
        return np.array(gradient)

    def H(self,x):
        hessian = np.zeros((2,2))
        hessian[0][0]=1200*x[0]**2-400*x[1]+2
        hessian[0][1]=-400*x[0]
        hessian[1][0]=-400*x[0]
        hessian[1][1]=200
        return hessian
    
    def r(self,x):
        r0=10*np.sqrt(2)*(x[1]-x[0]**2)
        r1=np.sqrt(2)*(1-x[0])
        return np.array([r0,r1])

    def J(self,x):
        jacobi = np.zeros((2,2))
        jacobi[0][0]=-20*np.sqrt(2)*x[0]
        jacobi[0][1]=10*np.sqrt(2)
        jacobi[1][0]=-np.sqrt(2)
        jacobi[1][1]=0
        return jacobi

    def step(self):
        fx=self.f(self.x0)
        gk=self.g(self.x0)
        self.log_loss.append(math.log(fx))
        self.t.append(time()-self.t0)
        self.iter.append(self.iters)
        self.iters+=1

        gk2=np.dot(gk,gk)
        if gk2<self.err**2:
            return True
        J=self.J(self.x0)
        JT=J.transpose()
        dk=-np.linalg.inv(JT@J)@(JT@self.r(self.x0))
        self.x0 = self.x0 + dk
        return False
    
    def opt(self):
        self.t0=time()
        while True:
            if self.step():
                break
    
    def plot_fi(self):
        plt.title("Gauss-Newton's Method")
        plt.xlabel('iterations')
        plt.ylabel('log(f(x)-p*)')
        plt.plot(self.iter,self.log_loss)
        plt.show()
    
    def plot_ft(self):
        plt.title("Gauss-Newton's Method")
        plt.xlabel('time')
        plt.ylabel('log(f(x)-p*)')
        plt.plot(self.t,self.log_loss)
        plt.show()

x0=np.array([-2,2])
err=0.00005
alpha=0.5
beta=0.5

dmn=DNM(x0,err,alpha,beta)
dmn.opt()
dmn.plot_fi()
dmn.plot_ft()

gmn=GNM(x0,err)
gmn.opt()
gmn.plot_fi()
gmn.plot_ft()