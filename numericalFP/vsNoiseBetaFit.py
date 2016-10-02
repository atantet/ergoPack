import os
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import ergoPlot

# Get model
model = 'Hopf'
gam = 1.
#betaRng = np.arange(0., 2.05, 0.05)
betaRng = np.arange(0., 1.5, 0.05)
epsRng = np.arange(.05, 2.05, .05)
mu = 0.

# Grid definition
dim = 2
nx0 = 200
nSTD = 5
nevPlot = 1
nPlot = 5

# Number of eigenvalues
nev = 21
#nev = 201

# Directories
resDir = '../results/numericalFP/%s' % model
plotDir = '../results/plot/numericalFP/%s' % model
os.system('mkdir %s 2> /dev/null' % plotDir)
mu += 1.e-8
if mu < 0:
    signMu = 'm'
else:
    signMu = 'p'

# Allocation
eigVal = np.empty((epsRng.shape[0], betaRng.shape[0], nev), dtype=complex)
colors = rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < epsRng.shape[0]:
    colors = np.concatenate((colors, rcParams['axes.prop_cycle'].by_key()['color']))

# Read eigenvalues
for ibeta in np.arange(betaRng.shape[0]):
    beta = betaRng[ibeta]
    beta += 1.e-8
    if beta < 0:
        signBeta = 'm'
    else:
        signBeta = 'p'
    for ieps in np.arange(epsRng.shape[0]):
        eps = epsRng[ieps]
        postfix = '_%s_mu%s%02d_beta%s%03d_eps%03d_nx%d_nSTD%d_nev%d' \
                  % (model, signMu, int(round(np.abs(mu) * 10)),
                     signBeta, int(round(np.abs(beta) * 100)), int(round(eps * 100)),
                     nx0, nSTD, nev)

        # Read eigenvalues
        eigValii = np.empty((nev,), dtype=complex)
        ergoPlot.loadtxt_complex('%s/eigValForward%s.txt' \
                                 % (resDir, postfix), eigValii)
        isort = np.argsort(-eigValii.real)
        eigValii = eigValii[isort]
        eigVal[ieps, ibeta] = eigValii

# Comparison with analytics of real part of the second eigenvalue versus the noise
# Regression model: \Re(\lambda_i) = a_i \epsilon + b_i \epsilon \beta^2)
#                                  = a_i \epsilon (1 + c_i \beta^2)
# with c_i = b_i / a_i
regModel = np.empty((nev, 2))
R2 = np.empty((nev,))
nbeta = betaRng.shape[0]
neps = epsRng.shape[0]
nReg = nbeta * neps
X = np.empty((nReg, 2))
for ieps in np.arange(neps):
    eps = epsRng[ieps]
    X[ieps*nbeta:(ieps+1)*nbeta, 0] = eps
    X[ieps*nbeta:(ieps+1)*nbeta, 1] = eps * betaRng**2
X = np.matrix(X)
iev = 1
ev = iev
regModel[0] = 0.
evRank = []
evRank.append(0)
EPS, BETA = np.meshgrid(epsRng, betaRng, indexing='ij')
while ev <= nevPlot:
    print 'Fitting for eigenvalue', ev
    eigv = eigVal[:, :, ev].flatten().real
    B = np.matrix(eigv).T
    A = (X.T * X)**(-1) * (X.T * B)
    Y = X * A
    Stot = np.var(eigv)
    Sres = np.sum((np.array(Y)[:, 0] - eigv)**2) / nReg
    R2[iev] = 1. - Sres / Stot
    regModel[iev] = np.array([A[0, 0], A[1, 0] / A[0, 0]])
    evRank.append(ev)
    if np.abs(eigVal[0, 0, ev].imag) > 1.e-6:
        ev += 1
    ev += 1
    iev += 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = np.array(Y)[:, 0].reshape(neps, nbeta)
    ax.plot_surface(EPS, BETA, Z, rstride=8, cstride=8, alpha=0.3)
regModel = regModel[:iev]
R2 = R2[:iev]
    
