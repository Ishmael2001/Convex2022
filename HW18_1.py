import numpy as np
from matplotlib import pyplot as plt

class PM:  #Penalty Method
    def __init__(self,A,b,x0):
        self.A=A
        self.b=b
        self.x0=x0
        self.maxiter=200
        self.minValue=self.f(A.T@np.linalg.inv(A@A.T)@b)
    def f(self,x):
        return np.linalg.norm(x, 2)/2
    def f_ab(self,gamma):
        return lambda x: np.linalg.norm(x, 2)**2/2 + gamma * np.linalg.norm(self.A@x-self.b)**2
    def g_ab(self,gamma):
        return lambda x: x + 2 * gamma * self.A.T@(self.A@x-self.b)
    def f_bc(self,gamma):
        return lambda x: np.linalg.norm(x, 2)**2/2 + gamma * np.sum((self.A@x-self.b)**4)
    def g_bc(self,gamma):
        return lambda x: x + 4 * gamma * self.A.T@((self.A@x-self.b)**3)
    def f_ab2(self,gamma):
        return lambda x: np.linalg.norm(x, 2)**2/2 + gamma * np.linalg.norm(self.A@x-self.b, 1)
    def g_ab2(self,gamma):
        return lambda x: x + gamma * self.A.T@np.sign(self.A@x-self.b)
    def lsearch(self,f,x,g,d, alpha=0.4, beta=0.8):
        t = 1.0
        value = f(x)
        while f(x + t*d) > value + alpha*t*np.dot(g, d):
            t *= beta
        return t
    def descent(self,f, x, grad):
        xn = x.copy()
        g = grad(xn)
        grad_norm = np.linalg.norm(g, 2)
        d = -g/grad_norm
        t = self.lsearch(f, xn, g, d)
        xn += t*d
        return xn, t
    def step(self,method):
        if method=="absolute value penalty function":
            self.time1 = []
            self.values1 = []
            self.pvalues1 = []
            timestep = 0
            x = self.x0.copy()
            gamma = 0
            count = 0 
            eps = 1e-10
            oldvalue = self.f(x)
            while True:
                value = self.f(x)
                print("Iteration:", timestep, "Value", value)
                if abs(value - oldvalue) < eps:
                    count += 1
                else:
                    count = 0
                oldvalue = value
                if timestep > self.maxiter or count >= 5:
                    break
                for i in range(20):
                    x, t = self.descent(self.f_ab(gamma), x, self.g_ab(gamma))
                print(self.f_ab(gamma)(x))
                self.time1.append(timestep)
                self.values1.append(value)
                self.pvalues1.append(self.f_ab(gamma)(x))
                gamma += 100
                timestep += 1
        elif method=="Courant-Beltrami penalty function":
            self.time2 = []
            self.values2 = []
            self.pvalues2 = []
            timestep = 0
            x = self.x0.copy()
            gamma = 0
            count = 0 
            eps = 1e-10
            oldvalue = self.f(x)
            maxIter = 200
            while True:
                value = self.f(x)
                print("Iteration:", timestep, "Value", value)
                if abs(value - oldvalue) < eps:
                    count += 1
                else:
                    count = 0
                oldvalue = value
                if timestep > self.maxiter or count >= 5:
                    break
                for i in range(10):
                    x, t = self.descent(self.f_bc(gamma), x, self.g_bc(gamma))  # 此时使用无穷范数
                self.time2.append(timestep)
                self.values2.append(value)
                self.pvalues2.append(self.f_bc(gamma)(x))
                gamma += 100
                timestep += 1
    def plot(self):
        plt.plot(self.time1, np.log(self.values1)-self.minValue)
        plt.plot(self.time2, np.log(self.values2)-self.minValue)
        plt.legend(['Absolute', 'Courant-Beltrami'])
        plt.xlabel("Iteration number $(k/100)$")
        plt.ylabel("$\log (f(\mathbf{x}_{k})-f^{*})$")
        plt.savefig('result1gamma100.png', dpi=200)
        plt.show()


x0 = np.random.rand(300)
A=np.random.normal(0, 1, (200, 300))
b=np.random.normal(0, 1, 200)
trial=PM(A,b,x0)
trial.step("absolute value penalty function")
trial.step("Courant-Beltrami penalty function")
trial.plot()