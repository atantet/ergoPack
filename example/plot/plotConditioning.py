import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import itertools
import pylibconfig2
import ergoPlot

configFile = '../cfg/OU2d.cfg'
#configFile = '../cfg/Battisti1989.cfg'
#configFile = '../cfg/Suarez1988.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

delayName = ""
for d in np.arange(len(cfg.model.delaysDays)):
    delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

L = cfg.simulation.LCut + cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt)
srcPostfix = "_%s%s_L%d_spinup%d_dt%d_samp%d" \
             % (cfg.model.caseName, delayName, L, cfg.simulation.spinup,
                -np.round(np.log10(cfg.simulation.dt)), printStepNum)

embedding = (np.array(cfg.observable.embeddingDays) / 365 \
             / cfg.simulation.printStep).astype(int)
dimObs = len(cfg.observable.components)
obsName = ""
for d in np.arange(dimObs):
    obsName = "%s_c%d_e%d" % (obsName, cfg.observable.components[d],
                              cfg.observable.embeddingDays[d])

N = np.prod(np.array(cfg.grid.nx))
gridPostfix = ""
for d in np.arange(dimObs):
    gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                    cfg.grid.nSTDLow[d], cfg.grid.nSTDHigh[d])
gridPostfix = "%s%s%s" % (srcPostfix, obsName, gridPostfix)
gridFile = '%s/grid/grid%s.txt' % (cfg.general.resDir, gridPostfix)

nLags = len(cfg.transfer.tauRng)

nevPlot = cfg.spectrum.nev
xminEigVal = -cfg.stat.rateMax
yminEigVal = -cfg.stat.angFreqMax
ev_xlabel = r'$x_1$'
ev_ylabel = r'$x_2$'

# Read grid
(X, Y) = ergoPlot.readGrid(ergoPlot.gridFile, ergoPlot.dimObs)
coord = (X.flatten(), Y.flatten())


eigenCondition = np.empty((nLags, cfg.spectrum.nev))
eigVal = np.empty((nLags, cfg.spectrum.nev), dtype=complex)
eigValGen = np.empty((nLags, cfg.spectrum.nev), dtype=complex)
for lag in np.arange(nLags):
    tau = cfg.transfer.tauRng[lag]
    
    # Define file names
    postfix = "%s_tau%03d" % (ergoPlot.gridPostfix, tau * 1000)
    EigValForwardFile = '%s/eigval/eigValForward_nev%d%s.txt' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfix)
    EigVecForwardFile = '%s/eigvec/eigVecForward_nev%d%s.txt' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfix)
    EigValBackwardFile = '%s/eigval/eigValBackward_nev%d%s.txt' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfix)
    EigVecBackwardFile = '%s/eigvec/eigVecBackward_nev%d%s.txt' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfix)

    # Read stationary distribution
    statDist = np.loadtxt('%s/transfer/initDist/initDist%s.txt' \
                          % (cfg.general.resDir, postfix))

    # Read transfer operator spectrum from file and create a bi-orthonormal basis
    # of eigenvectors and adjoint eigenvectors:
    print 'Readig spectrum of tau = ', tau
    (eigValForward, eigVecForward, eigValBackward, eigVecBackward) \
        = ergoPlot.readSpectrum(cfg.spectrum.nev, EigValForwardFile, EigVecForwardFile,
                                EigValBackwardFile, EigVecBackwardFile, statDist)
    # Save eigenvalues
    eigVal[lag] = eigValForward.copy()
    # Get generator eigenvalues
    eigValGen[lag] = (np.log(np.abs(eigValForward)) + np.angle(eigValForward)*1j) / tau
    # Get condition number
    eigenCondition[lag] = ergoPlot.getEigenCondition(eigVecForward, eigVecBackward,
                                                     statDist)

    print 'lag ', lag
    print eigVal[lag]
    print eigValGen[lag]
    print eigenCondition[lag]

    # Smoothen
    if lag > 0:
        tmpEigVal = eigVal[lag].copy()
        tmpEigValGen = eigValGen[lag].copy()
        tmpEigenCondition = eigenCondition[lag].copy()
        for ev in np.arange(cfg.spectrum.nev):
            idx = np.argmin(np.abs(tmpEigVal[ev] - eigVal[lag-1]))
            eigVal[lag, idx] = tmpEigVal[ev]
            eigValGen[lag, idx] = tmpEigValGen[ev]
            eigenCondition[lag, idx] = tmpEigenCondition[ev]
        print 'sort'
        print eigVal[lag]
        print eigValGen[lag]
        print eigenCondition[lag]

    plt.figure()
    plt.scatter(eigValGen[lag].real, eigValGen[lag].imag)
    plt.xlim(xminEigVal / 2, -xminEigVal / 100)
    plt.ylim(yminEigVal, -yminEigVal)



# Plot condition numbers versus absolute value of eigenvalue
# for different lags (and for each eigenvalue)
fig = plt.figure()
ax = fig.add_subplot(111)
markers = itertools.cycle((',', '+', '.', 'o', '*', '^', 'v', '<', '>', '8', 's'))
colors = itertools.cycle(('r', 'g', 'b', 'c', 'm', 'y', 'k'))
msize = 20
maxCond = 2.5
for ev in np.arange(nevPlot):
    color = colors.next()
    plt.plot(eigValGen[:, ev].real, eigenCondition[:, ev],
             linestyle='-', linewidth=1, color=color)
    while markers.next() != ',':
        continue
    for k in np.arange(nLags):
        marker = markers.next()
        plt.scatter(eigValGen[k, ev].real, eigenCondition[k, ev],
                    color=color, marker=marker, s=msize)
ax.set_xlim(xminEigVal / 2, -xminEigVal / 100)
ax.set_ylim(0., maxCond)
