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
        return x
    def update_E(self,Z,L,lam,beta):
        M = self.D - self.D@Z + L/beta
        Enew = self.H(M, lam/beta)
        return Enew
    def update_Z(self,Z, E, L, Y, M, l, beta, etaZ):
        N = self.D + L/beta - E
        W = Z - (self.D.T@(self.D@Z-N) + np.sum(l)/beta + np.sum(Z, axis=0)/beta-np.ones((1, 300)))/etaZ + Y/etaZ - M/beta/etaZ
        U, sigma, V = np.linalg.svd(W)
        sigma = self.S(sigma, 1/beta/etaZ)
        print()
        Znew = U @ np.diag(sigma) @ V
        return Znew
    def update_beta(self,beta, maxBeta, rho0, eta, eps2, dE, dZ, dY, normD):
        satisfied = False
        if beta*np.max((np.sqrt(3.1)*dE, np.sqrt(eta)*dZ, np.sqrt(3.1)*dY))/normD < eps2:
            rho = rho0
            satisfied = True
        else:
            rho = 1
        return min(beta*rho, maxBeta), satisfied
    def step(self,lam=0.1,beta=1e-4,maxBeta=1e4,rho0=1.9,eps1=1e-2,eps2=1e-2,r=5):
        step = 0
        p, n= self.D.shape
        E = np.zeros((p, n))
        Z = np.zeros((n, n))
        L = np.zeros((p, n))
        Y = np.zeros((n, n))
        M = np.zeros((n, n))
        l = np.zeros(n)
        dL = self.D - self.D@Z - E
        dLnorm = np.linalg.norm(dL)
        normD = np.linalg.norm(self.D)
        dM = Z - Y
        dMnorm = np.linalg.norm(dM)
        etaZ = normD**2*3.1
        crit1 = False
        crit2 = False
        while step < self.maxiter:
            if step % 1 == 0:
                value = self.f(Z, E, lam)
                print(step, value, 'gap', dLnorm, dMnorm, beta, crit1, crit2)
                self.every_step.append(value)
            self.every_step_beta.append(beta)
            Enew = self.update_E(Z, L, lam, beta)
            Znew = self.update_Z(Z, E, L, Y, M, l, beta, etaZ)
            Ynew = np.maximum(Z, 0)
            dE = np.linalg.norm(Enew-E)
            dZ = np.linalg.norm(Znew-Z)
            dY = np.linalg.norm(Ynew-Y)
            E = Enew
            Z = Znew
            Y = Ynew
            dl = np.sum(Z, axis=0) - np.ones((1, n))
            dL = -self.D@Z - E + self.D
            dLnorm = np.linalg.norm(dL)
            dM = Z - Y
            dMnorm = np.linalg.norm(dM)
            L = L + beta * dL
            l = l + beta * dl
            M = M + beta * dM
            crit1 = dLnorm/normD < eps1
            beta, crit2 = self.update_beta(
                beta, maxBeta, rho0, etaZ, eps2, dE, dZ, dY, normD)
            if crit1 and crit2:
                print("Converged at step", step)
                value = self.f(Z, E, lam)
                print(step, value, 'gap', dLnorm, dMnorm)
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