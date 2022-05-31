import numpy as np
import matplotlib.pyplot as plt
import time

class MM:
	def __init__(self,x,A,lamda,b,err,method,miu=0.5):
		self.x=x
		self.A=A
		self.lamda=lamda
		self.b=b
		self.method=method
		self.err=err
		self.miu=miu
		self.fk=[]
		self.time=[]
		self.iters=[]
		self.iter=0
	
	def f(self,x):
		temp = self.A*x - self.b
		return (temp.transpose()*temp * 0.5 + self.lamda * np.sum(np.abs(x)))[0,0]
	
	def g(self,x):  #定义梯度和次梯度
		c = self.A.transpose() * (self.A * x - self.b)
		for i in range(c.shape[0]):
			if x[i] == 0:
				if abs(c[i]) <= self.lamda:
					c[i] = 0
				elif c[i] > self.lamda:
					c[i] = c[i] - self.lamda
				elif c[i] < -self.lamda:
					c[i] = c[i] + self.lamda
			else:
				c[i] += self.lamda*np.sign(x[i])
		return c
	
	def step_1(self):
		a=1
		c = self.A.transpose() * (self.A * self.x - self.b)
		while True:
			x_temp=np.zeros_like(self.x)
			for i in range(self.x.shape[0]):
				x_1 = self.x[i] - a*c[i] - a* self.lamda
				x_2 = self.x[i] - a*c[i] + a* self.lamda
				if x_1 > 0:
					x_temp[i] = x_1
				elif x_2 < 0:
					x_temp[i] = x_2
				else:
					x_temp[i] = 0
			if self.f(x_temp) < self.f(self.x):
				break
			a *= self.miu
		self.x=x_temp
		self.iters.append(self.iter)
		self.iter+=1
		self.fk.append(self.f(self.x))
		self.time.append(time.time() - self.t0)
		g=self.g(self.x)
		if (g.transpose()*g)[0,0]<self.err**2:
			return True
		else:
			return False
	
	def step_2(self):
		self.x=np.linalg.inv(self.A.transpose()*self.A+self.lamda*np.diag(1/np.abs(self.x.transpose().A[0])))*self.A.transpose()*self.b
		self.iters.append(self.iter)
		self.iter+=1
		self.fk.append(self.f(self.x))
		self.time.append(time.time() - self.t0)
		g=self.g(self.x)
		if (g.transpose()*g)[0,0]<self.err**2:
			return True
		else:
			return False
	

	def opt(self):
		self.t0 = time.time()
		self.time.append(time.time() - self.t0)
		self.iters.append(self.iter)
		self.fk.append(self.f(self.x))
		self.iter+=1
		if self.method=="1":
			while True:
				if self.step_1():
					break
		if self.method=="2":
			while True:
				if self.step_2():
					break
	
	def plot_fi(self):
		plt.title("MM algorithm, method={}".format(self.method))
		plt.xlabel('iterations')
		plt.ylabel('f(x)')
		plt.plot(self.iters,self.fk)
		plt.show()
	
	def plot_ft(self):
		plt.title("MM algorithm, method={}".format(self.method))
		plt.xlabel('time')
		plt.ylabel('f(x)')
		plt.plot(self.time,self.fk)
		plt.show()

x0 = np.mat(np.random.randn(3)).transpose()
A = np.mat(np.random.randn(3,3))
b = np.mat(np.random.randn(3)).transpose()
lamda=1

for i in range(2):
	x=np.zeros_like(x0)
	for j in range(x0.shape[0]):
		x[j]=x0[j]
	mm=MM(x,A,1,b,0.01,str(i+1))
	mm.opt()
	mm.plot_fi()
	mm.plot_ft()

