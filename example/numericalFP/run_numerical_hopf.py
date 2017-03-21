import os
import numpy as np

# Python script for numerical Fokker-Planck
scriptFile = 'numericalFP_Hopf_cmd.py'

# # Experiment 1: mu around 0, beta = 0.5,
# # other parameters fixed and non-vanishing
# muRng = np.arange(-5., 5.1, 0.1)
# betaRng = np.array([0.5])
# gammaRng = np.array([1.])
# epsRng = np.array([.25])
# #epsRng = np.array([.5])
# #epsRng = np.array([1.])
# #epsRng = np.array([1.5])
# #epsRng = np.array([2.])
# #epsRng = np.array([3.])
# #epsRng = np.array([4.])
# #epsRng = np.array([10.])
# nev = 21
# nx0 = 200
# saveEigVecForward = 0
# saveEigVecBackward = 0

# # Experiment 2: Same as 1 but for only a few mu and many eigenvalues
# muRng = np.array([-5., 0., 3., 7.])
# betaRng = np.array([0.5])
# gammaRng = np.array([1.])
# epsRng = np.array([1.])
# nev = 201
# nx0 = 200
# saveEigVecForward = 1
# saveEigVecBackward = 1

# # Experiment 3: Same as 2 but with beta = 0.
# muRng = np.array([-5., 0., 3., 7.])
# betaRng = np.array([0.])
# gammaRng = np.array([1.])
# epsRng = np.array([1.])
# nev = 201
# nx0 = 200
# saveEigVecForward = 1
# saveEigVecBackward = 1

# # Experiment 4: Varying eps and beta for mu = 0
# # and other parameters fixed and non-vanishing
# muRng = np.array([0.])
# betaRng = np.arange(0., 2.05, 0.05)
# gammaRng = np.array([1.])
# epsRng = np.arange(0.05, 2.05, 0.05)
# nev = 21
# nx0 = 200
# saveEigVecForward = 0
# saveEigVecBackward = 0


# # Experiment 5: Varying eps for mu = 5 and other parameters fixed and non-vanishing
# muRng = np.array([5.])
# betaRng = np.array([0.5])
# gammaRng = np.array([1.])
# epsRng = np.arange(0.05, 2.05, 0.05)
# nev = 21
# nx0 = 200
# saveEigVecForward = 0
# saveEigVecBackward = 0

# # Experiment 6: Varying beta for mu = 0 and other parameters fixed and non-vanishing
# muRng = np.array([0.])
# betaRng = np.arange(0., 2.05, 0.05)
# gammaRng = np.array([1.])
# epsRng = np.array([1.])
# nev = 21
# nx0 = 200
# saveEigVecForward = 0
# saveEigVecBackward = 0

# # Experiment 7: Same as 4 buth with larger nx0
# muRng = np.array([0.])
# betaRng = np.array([0.5])
# gammaRng = np.array([1.])
# #epsRng = np.arange(0.05, 2.05, 0.05)
# epsRng = np.arange(0.4, 2.05, 0.05)
# nev = 21
# nx0 = 300
# # nx0 = 400
# saveEigVecForward = 0
# saveEigVecBackward = 0

# Experiment 8: mu around 0, beta = 1.,
# other parameters fixed and non-vanishing
muRng = np.arange(-1., 1.1, 0.1)
betaRng = np.array([1.])
gammaRng = np.array([1.])
#epsRng = np.array([.25])
#epsRng = np.array([.5])
#epsRng = np.array([1.])
#epsRng = np.array([1.5])
#epsRng = np.array([2.])
#epsRng = np.array([3.])
epsRng = np.array([4.])
#epsRng = np.array([10.])
nev = 21
nx0 = 200
saveEigVecForward = 0
saveEigVecBackward = 0

# Numerical analysis parameters
nSTD = 5
tol = 1.e-6

for mu in muRng:
    for beta in betaRng:
        for gamma in gammaRng:
            for eps in epsRng:
                os.system('python %s %s %s %s %s %s %s %s %s %s %s' \
                          % (scriptFile, mu, beta, gamma, eps, nx0, nSTD,
                             nev, tol, saveEigVecForward, saveEigVecBackward))

