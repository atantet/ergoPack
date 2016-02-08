import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import ergoPlot

#configFile = '../cfg/OU2d.cfg'
configFile = '../cfg/Battisti1989.cfg'
#configFile = '../cfg/Suarez1988.cfg'
ergoPlot.readConfig(configFile)

nevPlot = 15
xminEigVal = -ergoPlot.rateMax
yminEigVal = -ergoPlot.angFreqMax
ev_xlabel = r'$x_1$'
ev_ylabel = r'$x_2$'
nComponents = 15

# Read grid
(X, Y) = ergoPlot.readGrid(ergoPlot.gridFile, ergoPlot.dimObs)
coord = (X.flatten(), Y.flatten())


eigenCondition = np.empty((ergoPlot.nLags, ergoPlot.nev))
eigValGen = np.empty((ergoPlot.nLags, ergoPlot.nev), dtype=complex)
for lag in np.arange(ergoPlot.nLags):
    tau = ergoPlot.tauRng[lag]
    
    # Define file names
    postfix = "%s_tau%03d" % (ergoPlot.gridPostfix, tau * 1000)
    EigValFile = '%s/eigVal/eigVal_nev%d%s.txt' % (ergoPlot.specDir, ergoPlot.nev, postfix)
    EigVecFile = '%s/eigVec/eigVec_nev%d%s.txt' % (ergoPlot.specDir, ergoPlot.nev, postfix)
    EigValAdjointFile = '%s/eigVal/eigValAdjoint_nev%d%s.txt' \
                        % (ergoPlot.specDir, ergoPlot.nev, postfix)
    EigVecAdjointFile = '%s/eigVec/eigVecAdjoint_nev%d%s.txt' \
                        % (ergoPlot.specDir, ergoPlot.nev, postfix)

    # Read stationary distribution
    statDist = np.loadtxt('%s/transfer/initDist/initDist%s.txt' % (ergoPlot.resDir, postfix))

    # Read transfer operator spectrum from file and create a bi-orthonormal basis
    # of eigenvectors and adjoint eigenvectors:
    print 'Readig spectrum of tau = ', tau
    (eigVal, eigVec, eigValAdjoint, eigVecAdjoint) \
        = ergoPlot.readSpectrum(ergoPlot.nev, EigValFile, EigVecFile,
                                EigValAdjointFile, EigVecAdjointFile, statDist)

    print 'Getting conditionning of eigenvectors...'
    eigenCondition[lag] = ergoPlot.getEigenCondition(eigVec, eigVecAdjoint, statDist)
    numpy.set_printoptions(precision=2)
    print "Eigen condition numbers:", eigenCondition[lag]

    # Get generator eigenvalues
    print 'Getting generator eigenvalues...'
    eigValGen[lag] = (np.log(np.abs(eigVal)) + np.angle(eigVal)*1j) / tau

    # Smoothen
    if lag > 0:
        tmp = eigValGen[lag].copy()
        for ev in np.arange(ergoPlot.nev):
            eigValGen[lag, ev] = tmp[np.argmin(np.abs(tmp - eigValGen[lag-1, ev]))]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
for ev in np.arange(nevPlot):
    eig = np.array([eigValGen[:, ev].real, eigValGen[:, ev].imag]).T.reshape(-1, 1, 2)
    segments = np.concatenate((eig[:-1], eig[1:]), 1)
    lc = LineCollection(segments, cmap=plt.get_cmap('hot_r'),
                        norm=plt.Normalize(ergoPlot.tauRng[0], ergoPlot.tauRng[-1]))
    #ax.plot(eigValGen.real[:, ev], eigValGen.imag[:, ev])
    lc.set_array(ergoPlot.tauRng)
    lc.set_linewidth(3)
    ax.add_collection(lc)
    plt.plot(eigValGen[0, ev].real, eigValGen[0, ev].imag, 'ok')
    plt.plot(eigValGen[-1, ev].real, eigValGen[-1, ev].imag, '>k')
ax.set_xlim(xminEigVal, -xminEigVal / 100)
ax.set_ylim(yminEigVal, -yminEigVal)

fig = plt.figure()
ax = fig.add_subplot(111)
for ev in np.arange(nevPlot):
    ax.plot(ergoPlot.tauRng, eigenCondition[:, ev])
ax.set_ylim(0., 10.)
