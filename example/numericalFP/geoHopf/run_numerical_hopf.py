import os
import numpy as np

# Python script for numerical Fokker-Planck
scriptFile = 'numFP_geoHopf_cmd.py'

# Experiment 2: Same as 1 but for only a few mu and many eigenvalues
muRng = np.array([-5.])
#muRng = np.array([7.])

betaRng = np.array([0.])
#betaRng = np.array([0.5])

#gammaRng = np.array([0.5])
gammaRng = np.array([1.])

deltaRng = np.array([0., 0.25, 0.5, 0.75, 1.])

epsRng = np.array([1.])

nev = 21

nx0 = 100
#nx0 = 200

saveEigVecForward = 1
saveEigVecBackward = 1

# Numerical analysis parameters
nSTD = 5
tol = 1.e-6

for mu in muRng:
    for beta in betaRng:
        for gamma in gammaRng:
            for delta in deltaRng:
                for eps in epsRng:
                    os.system('python %s %s %s %s %s %s %s %s %s %s %s %s' \
                              % (scriptFile, mu, beta, gamma, delta, eps, nx0, nSTD,
                                 nev, tol, saveEigVecForward, saveEigVecBackward))

