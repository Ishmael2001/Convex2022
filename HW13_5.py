import numpy as np
import math
import matplotlib.pyplot as plt

class DFP: 
    def __init__(self,x0,err,alpha,beta):
        self.x0=x0
        self.err=err
        self.alpha=alpha
        self.beta=beta
        self.f_iter=[]
        self.iters=0
        self.iter=[]
        self.buchang=[]
        self.dk=0
        self.Hk=np.mat([[1,0,0],[0,1,0],[0,0,1]])
        self.Hk_pd=True
        
    def f(self,x):
        ans=(3-x[0,0])**2+7*(x[1,0]-x[0,0]**2)**2+9*(x[2,0]-x[0,0]-x[1,0]**2)**2
        return ans

    def g(self,x):
        gradient=[]
        gradient.append(2*(x[0,0]-3)+28*(x[0,0]**2-x[1,0])*x[0,0]+18*(x[0,0]+x[1,0]**2-x[2,0]))
        gradient.append(-14*(x[0,0]**2-x[1,0])+36*(x[0,0]+x[1,0]**2-x[2,0])*x[1,0])
        gradient.append(-18*(x[0,0]+x[1,0]**2-x[2,0]))
        return np.mat(gradient).transpose()
    
    def step(self):       
        fk=self.f(self.x0)
        gk=self.g(self.x0)
        if (gk.transpose()*gk)[0,0]<self.err**2:
            return True
        self.dk=np.mat(-(self.Hk*gk))
        self.f_iter.append(fk)
        self.iter.append(self.iters)
        self.iters+=1
        t=1
        while self.f(self.x0+t*self.dk)>self.f(self.x0)+self.alpha*t*(gk.transpose()*self.dk) and t>0.001:
            t*=self.beta
        self.x0=self.x0+t*self.dk
        self.buchang.append(t)
        delta_x=t*self.dk
        delta_g=self.g(self.x0)-gk
        self.Hk=self.Hk+(delta_x*delta_x.transpose())/(delta_x.transpose()*delta_g)-(self.Hk*delta_g)*((self.Hk*delta_g).transpose())/(delta_g.transpose()*self.Hk*delta_g)
        B=np.linalg.eigvals(self.Hk)
        if (np.all(B)>0)==False:
            self.Hk_pd=False
        return False
    
    def opt(self):
        while True:
            print(self.x0[2,0])
            if self.step():
                break
        print(self.x0)
    
    def plot_fi(self):
        plt.title("DFP algorithm")
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.plot(self.iter,self.f_iter)
        plt.show()

    def plot_ft(self):
        plt.title("DFP algorithm")
        plt.xlabel('iterations')
        plt.ylabel('a')
        plt.plot(self.iter,self.buchang)
        plt.show()

x0=np.mat([0,0,0]).transpose()
err=0.005
alpha=0.5
beta=0.5
dfp1=DFP(x0,err,alpha,beta)
dfp1.opt()
dfp1.plot_fi()
dfp1.plot_ft()
print(dfp1.Hk_pd)
