import numpy as np
from matplotlib import pyplot as plt
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

class LADMAP:
    def __init__(self,D):
        self.D=D
        self.maxiter=30000
        self.every_step=[]
        self.every_step_beta=[]
    def norm21(self,x):
        return np.linalg.norm(np.linalg.norm(x, 2, axis=0), 1)
    def f(self,Z,E,lam):
        return np.linalg.norm(Z,'nuc') + lam*self.norm21(E)
    def H(self,M,eps):
        norm = np.linalg.norm(M, 2, axis=0)
        M[:, norm>eps] *= (norm[norm>eps] - eps)/norm[norm>eps]
        M[:, norm<=eps] = 0
        return M
    def S(self,x,eps):
        count = np.sum([x >= eps])
        x[np.abs(x) < eps] = 0
        x[x >= eps] -= eps
        x[x <= -eps] += eps
        return x, count
    def update_E(self,Z,L,lam,beta):
        M = self.D - self.D@Z + L/beta
        Enew = self.H(M, lam/beta)
        Enew[-1] = 0
        return Enew
    def update_Z(self,Z, E, L, beta, eta, r):
        N = self.D + L/beta - E
        W = Z - (self.D.T@(D@Z-N))/eta
        U, sigma, V = np.linalg.svd(W)
        sigma, count = self.S(sigma, 1/beta/eta)
        Znew = U@np.diag(sigma)@V
        return Znew, r
    def update_L(self,L, dL, beta):
        return L + beta * dL
    def update_beta(self,beta, maxBeta, rho0, eta, eps2, dE, dZ, normD):
        satisfied = False
        if beta*max(np.sqrt(2.1)*dE, np.sqrt(eta)*dZ)/normD < eps2:
            rho = rho0
            satisfied = True
        else:
            rho = 1
        return min(beta*rho, maxBeta), satisfied
    def step(self,lam=0.1,beta=1e-4,maxBeta=1e4,rho0=1.9,eps1=1e-3,eps2=1e-3,r=5):
        step = 0
        p, n= self.D.shape
        E = np.zeros((p, n))
        Z = np.zeros((n, n))
        L = np.zeros((p, n))
        dL = self.D - self.D@Z - E
        dLnorm = np.linalg.norm(dL)
        normD = np.linalg.norm(self.D)
        eta = normD**2.1
        while step < self.maxiter:
            if step % 100 == 0:
                value = self.f(Z, E, lam)
                print(step, value, 'gap', dLnorm)
                self.every_step.append(value)
            self.every_step_beta.append(beta)
            Enew = self.update_E(Z, L, lam, beta)
            Znew, r = self.update_Z(Z, Enew, L, beta, eta, r)
            dE = np.linalg.norm(Enew-E)
            dZ = np.linalg.norm(Znew-Z)
            E = Enew
            Z = Znew
            dl = np.sum(Z, axis=0) - np.ones((1, n))
            dL = self.D - self.D@Z - E
            dLnorm = np.linalg.norm(dL)
            L = self.update_L(L, dL, beta)
            crit1 = dLnorm/normD < eps1
            beta, crit2 = self.update_beta(
                beta, maxBeta, rho0, eta, eps2, dE, dZ, normD)
            if crit1 and crit2:
                print("Converged at step", step)
                value = self.f(Z, E, lam)
                print(step, value, 'gap', dLnorm)
                break
            step += 1
        return Z, E, L
    def plot(self):
        plt.plot(self.every_step)
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.show()
        plt.plot(self.every_step_beta)
        plt.xlabel('Steps')
        plt.ylabel('Beta')
        plt.show()


D = np.random.normal(0, 1, (200, 300))
D = np.vstack((D, np.ones(300)))
trial=LADMAP(D)
trial.step()
trial.plot()