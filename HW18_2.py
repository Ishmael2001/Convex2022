import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

class FW:  #Frank-Wolfe Algorithm
    def __init__(self,D,y,x0,norm):
        self.D=D
        self.y=y
        self.x=x0
        self.norm=norm  #取 1 or inf
        self.maxiter=30000
        self.every_step=[]
    def f(self,x):
        return np.linalg.norm(self.y-self.D @ x)**2
    def g(self,x):
        return 2*self.D.T @ (self.D @ x-self.y)
    def step(self):
        if self.norm=="1":
            cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.linalg.norm(x, 1)})
            res = minimize(self.f, self.x, constraints=cons, tol=1e-4, options={'maxiter': 1e4, 'disp': True})
            self.minValue = res.fun
        elif self.norm=="inf":
            cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.linalg.norm(x, np.inf)})
            res = minimize(self.f, self.x, constraints=cons, tol=1e-4, options={'maxiter': 1e4, 'disp': True})
            self.minValue = res.fun
        for i in range(self.maxiter*5):
            value = self.f(self.x)
            print(i, "th iteration, f(x)=", value)
            self.every_step.append(value-self.minValue)
            gamma = 2/(i+2)
            g = self.g(self.x)
            if self.norm == "1":
                d = np.argmax(np.abs(g))
                self.x = (1-gamma)*self.x
                self.x[d] -= gamma * np.sign(g[d])
            elif self.norm == 'inf':
                d = -np.sign(g)
                self.x += gamma * (d-self.x)
    def plot(self):
        if self.norm=="inf":
            plt.plot(np.log(self.every_step))
            plt.xlabel('Iterations')
            plt.ylabel('$\ln (f(x_k)-f^*)$')
            plt.savefig('wolfeInf')
            plt.show()
        elif self.norm=="1":
            plt.plot(np.log(self.every_step))
            plt.xlabel('Iterations')
            plt.ylabel('$\ln (f(x_k)-f^*)$')
            plt.savefig('wolfe1')
            plt.show()

class PD:  #Project Descend
    def __init__(self,D,y,x0,norm):
        self.D=D
        self.y=y
        self.x=x0
        self.norm=norm  #取 1 or inf
        self.every_step=[]
    def f(self,x):
        return np.linalg.norm(self.y-self.D @ x)**2
    def g(self,x):
        return 2*self.D.T @ (self.D @ x-self.y)
    def P(self,x):  #取投影
        if self.norm=="1":
            norm = np.linalg.norm(x, 1)
            if norm > 1:
                return x/norm
            return x
        elif self.norm=="inf":
            t = np.minimum(x, np.ones(300))
            return np.maximum(t, -np.ones(300))
    def l_search(self, d, alpha=0.4, beta=0.8):
        t = 1.0
        value = self.f(self.x)
        while self.f(self.x + t*d) > value + alpha*t*np.dot(self.g(self.x), d):
            t *= beta
        return t
    def projectedDescent(self):
        g = self.g(self.x)
        grad_norm = np.linalg.norm(g, 2)
        d = -g/grad_norm
        t = self.l_search(d)
        self.x += t*d
        return self.P(self.x)
    def step(self):
        if self.norm=="1":
            cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.linalg.norm(x, 1)})
            res = minimize(self.f, self.x, constraints=cons, tol=1e-4, options={'maxiter': 1e4, 'disp': True})
            self.minValue = res.fun
        elif self.norm=="inf":
            cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.linalg.norm(x, np.inf)})
            res = minimize(self.f, self.x, constraints=cons, tol=1e-4, options={'maxiter': 1e4, 'disp': True})
            self.minValue = res.fun
        steps = 0
        count = 0 
        err = 1e-13
        oldvalue = self.f(self.x)
        maxIter = 200000  # 最大迭代次数
        while True:
            value = self.f(self.x)
            print("Iteration:", steps, "Value", value)
            # 用函数值改变量作为终止条件
            if abs(value - oldvalue) < err:
                count += 1
            else:
                count = 0
            oldvalue = value
            self.every_step.append(value-self.minValue)
            if steps > maxIter or count >= 5:
                break
            self.x= self.projectedDescent()
            steps += 1
    def plot(self):
        if self.norm=="inf":
            plt.plot(np.log(self.every_step))
            plt.xlabel('Iterations')
            plt.ylabel('$\ln (f(x_k)-f^*)$')
            plt.savefig('project_descent_Inf')
            plt.show()
        elif self.norm=="1":
            plt.plot(np.log(self.every_step))
            plt.xlabel('Iterations')
            plt.ylabel('$\ln (f(x_k)-f^*)$')
            plt.savefig('project_descent_Inf1')
            plt.show()


x0 = np.random.rand(300)
D=np.random.normal(0, 1, (200, 300))
y=np.random.normal(0, 1, 200)
'''trial_1=FW(D,y,x0,"1")
trial_1.step()
trial_1.plot()
trial_inf=FW(D,y,x0,"inf")
trial_inf.step()
trial_inf.plot()'''

trial1_1=PD(D,y,x0,"1")
trial1_1.step()
trial1_1.plot()
trial1_inf=PD(D,y,x0,"inf")
trial1_inf.step()
trial1_inf.plot()

