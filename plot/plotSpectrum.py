import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import pylibconfig2
import ergoPlot

#configFile = '../cfg/OU2d.cfg'
#compName1 = 'x_1'
#compName2 = 'x_2'
#configFile = '../cfg/Battisti1989.cfg'
#configFile = '../cfg/Suarez1988.cfg'
#compName1 = r'y(t\prime)'
#compName2 = r'y(t\prime - \tau_d)'
configFile = '../cfg/Lorenz63.cfg'
compName1 = 'x'
compName2 = 'y'
compName3 = 'z'

cfg = pylibconfig2.Config()
cfg.read_file(configFile)

readSpectrum = ergoPlot.readSpectrum
makeBiorthonormal = ~cfg.spectrum.makeBiorthonormal
#readSpectrum = ergoPlot.readSpectrumCompressed
#makeBiorthonormal = True

tau = cfg.transfer.tauRng[0]

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

L = cfg.simulation.LCut + cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt)
caseName = cfg.model.caseName
if (hasattr(cfg.model, 'rho') & hasattr(cfg.model, 'sigma') & hasattr(cfg.model, 'beta')):
    caseName = "%s_rho%d_sigma%d_beta%d" \
               % (caseName, (int) (cfg.model.rho * 1000),
                  (int) (cfg.model.sigma * 1000), (int) (cfg.model.beta * 1000))
srcPostfix = "_%s%s_L%d_spinup%d_dt%d_samp%d" \
             % (caseName, delayName, L, cfg.simulation.spinup,
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
    if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.grid.nSTDLow[d], cfg.grid.nSTDHigh[d])
    else:
        gridPostfix = "%s_n%dminmax" % (gridPostfix, cfg.grid.nx[d])
gridPostfix = "%s%s%s" % (srcPostfix, obsName, gridPostfix)
gridFile = '%s/grid/grid%s.txt' % (cfg.general.resDir, gridPostfix)

nLags = len(cfg.transfer.tauRng)
ev_xlabel = r'$%s$' % compName1
ev_ylabel = r'$%s$' % compName2
corrLabel = r'$C_{%s, %s}(t)$' % (compName1, compName1)
powerLabel = r'$S_{%s, %s}(\omega)$' % (compName1, compName1)

nevPlot = 0
xminEigVal = -cfg.stat.rateMax
yminEigVal = -cfg.stat.angFreqMax
#plotBackward = False
plotBackward = True
#plotImag = False
plotImag = True
xlimEig = [xminEigVal, -xminEigVal/100]
ylimEig = [yminEigVal, -yminEigVal]
zlimEig = [cfg.stat.powerMin, cfg.stat.powerMax]
xticks = None
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))
zticks = np.logspace(np.log10(zlimEig[0]), np.log10(zlimEig[1]),
                     int(np.round(np.log10(zlimEig[1]/zlimEig[0]) + 1)))
#maxCondition = 5
maxCondition = 10


# # Read grid
coord = ergoPlot.readGrid(gridFile, dimObs)
if dimObs == 1:
    X = coord[0]
elif dimObs == 2:
    X, Y = np.meshgrid(coord[0], coord[1])
    coord = (X.flatten(), Y.flatten())
elif dimObs == 3:
    X, Y, Z = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
    coord = (X.T.flatten(), Y.T.flatten(), Z.T.flatten())

# # Define file names
# postfix = "%s_tau%03d" % (gridPostfix, tau * 1000)
# EigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.txt' \
#                     % (cfg.general.specDir, cfg.spectrum.nev, postfix)
# EigVecForwardFile = '%s/eigvec/eigvecForward_nev%d%s.txt' \
#                     % (cfg.general.specDir, cfg.spectrum.nev, postfix)
# EigValBackwardFile = '%s/eigval/eigvalBackward_nev%d%s.txt' \
#                     % (cfg.general.specDir, cfg.spectrum.nev, postfix)
# EigVecBackwardFile = '%s/eigvec/eigvecBackward_nev%d%s.txt' \
#                     % (cfg.general.specDir, cfg.spectrum.nev, postfix)

# # Read stationary distribution
# statDist = np.loadtxt('%s/transfer/initDist/initDist%s.txt' % (cfg.general.resDir, gridPostfix))

# # Read transfer operator spectrum from file and create a bi-orthonormal basis
# # of eigenvectors and backward eigenvectors:
# print 'Readig spectrum...'
# (eigValForward, eigVecForward, eigValBackward, eigVecBackward) \
#     = readSpectrum(EigValForwardFile, EigVecForwardFile,
#                    EigValBackwardFile, EigVecBackwardFile,
#                    statDist, makeBiorthonormal=makeBiorthonormal)


# print 'Getting conditionning of eigenvectors...'
# eigenCondition = ergoPlot.getEigenCondition(eigVecForward, eigVecBackward, statDist)

# # Get generator eigenvalues
# eigValGen = (np.log(np.abs(eigValForward)) + np.angle(eigValForward)*1j) / tau


# # Plot eigenvectors of transfer operator
# alpha = 0.01
# for ev in np.arange(nevPlot):
#     print 'Plotting real part of eigenvector %d...' % (ev + 1,)
#     ergoPlot.plot2D(X, Y, eigVecForward[:, ev].real, ev_xlabel, ev_ylabel, alpha)
#     dstFile = '%s/spectrum/eigvec/eigvecForwardReal_nev%d_ev%03d%s.%s' \
#               % (cfg.general.plotDir, cfg.spectrum.nev, ev + 1, postfix, ergoPlot.figFormat)
#     plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    
#     if plotImag & (eigValForward[ev].imag != 0):
#         print 'Plotting imaginary  part of eigenvector %d...' % (ev + 1,)
#         ergoPlot.plot2D(X, Y, eigVecForward[:, ev].imag, ev_xlabel, ev_ylabel, alpha)
#         dstFile = '%s/spectrum/eigvec/eigvecForwardImag_nev%d_ev%03d%s.%s' \
#                   % (cfg.general.plotDir, cfg.spectrum.nev, ev + 1, postfix, ergoPlot.figFormat)
#         plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    
#     # Plot eigenvectors of backward operator
#     if plotBackward:
#         print 'Plotting real part of backward eigenvector %d...' % (ev + 1,)
#         ergoPlot.plot2D(X, Y, eigVecBackward[:, ev].real, ev_xlabel, ev_ylabel, alpha)
#         dstFile = '%s/spectrum/eigvec/eigvecBackwardReal_nev%d_ev%03d%s.%s' \
#                   % (cfg.general.plotDir, cfg.spectrum.nev, ev + 1, postfix, ergoPlot.figFormat)
#         plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
        
#         if plotImag & (eigValForward[ev].imag != 0):
#             print 'Plotting imaginary  part of backward eigenvector %d...' % (ev + 1,)
#             ergoPlot.plot2D(X, Y, eigVecBackward[:, ev].imag, ev_xlabel, ev_ylabel, alpha)
#             dstFile = '%s/spectrum/eigvec/eigvecBackwardImag_nev%d_ev%03d%s.%s' \
#                       % (cfg.general.plotDir, cfg.spectrum.nev, ev + 1, postfix, ergoPlot.figFormat)
#             plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

            
# Define observables
corrName = 'C%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
powerName = 'S%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
f = coord[cfg.stat.idxf]
g = coord[cfg.stat.idxg]
# corrLabel = r'$C_{x_%d, x_%d}(t)$' % (cfg.stat.idxf + 1,
#                                       cfg.stat.idxg + 1)
# powerLabel = r'$S_{x_%d, x_%d}(\omega)$' % (cfg.stat.idxf + 1,
#                                             cfg.stat.idxg + 1)
realLabel = r'$\Re(\bar{\lambda}_k)$'
imagLabel = r'$\Im(\bar{\lambda}_k)$'

# Read ccf
print 'Reading correlation function and periodogram...'
corrSample = np.loadtxt('%s/correlation/%s_lag%d%s.txt'\
                        % (cfg.general.resDir, corrName, int(cfg.stat.lagMax),
                           srcPostfix))
lags = np.loadtxt('%s/correlation/lags_lag%d%s.txt'\
                  % (cfg.general.resDir, int(cfg.stat.lagMax),
                     srcPostfix))
powerSample = np.loadtxt('%s/power/%s_chunk%d%s.txt'\
                         % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
                            srcPostfix))
freq = np.loadtxt('%s/power/freq_chunk%d%s.txt' \
                  % (cfg.general.resDir, cfg.stat.chunkWidth,
                     srcPostfix))

# Convert to angular frequencies and normalize by covariance
angFreq = freq * 2*np.pi
cfg0 = ((f - (f * statDist).sum()) * statDist * (g - (g * statDist).sum())).sum()
powerSample /= 2 * np.pi * cfg0

# Reconstruct correlation and power spectrum
# Get normalized weights
weights = ergoPlot.getSpectralWeights(f, g, eigVecForward, eigVecBackward,
                                      statDist, skipMean=True)
# Remove components with heigh condition number
weights[eigenCondition > maxCondition] = 0.
eigenCondition[eigenCondition > maxCondition] = maxCondition
(corrRec, compCorrRec) = ergoPlot.spectralRecCorrelation(lags, f, g, eigValGen, weights,
                                                         statDist, skipMean=True, norm=True)
(powerRec, compPowerRec) = ergoPlot.spectralRecPower(angFreq, f, g, eigValGen, weights,
                                                     statDist, norm=True)

# Plot correlation reconstruction
ergoPlot.plotRecCorrelation(lags, corrSample, corrRec, plotPositive=True,
                            ylabel=corrLabel)
plt.savefig('%s/spectrum/reconstruction/%sRec_lag%d_nev%d%s.%s'\
            % (cfg.general.plotDir, corrName, int(cfg.stat.lagMax),
               cfg.spectrum.nev, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# PLot spectrum, powerSampledogram and spectral reconstruction
weights /= cfg0
msize = np.zeros((weights.shape[0]))
msize[weights.real > 0] = np.log10(weights[weights.real > 0].real)
msize[weights.real > 0] = (msize[weights.real > 0] + 8) * 10
# msize[weights.real > 0] = (msize[weights.real > 0] + 6) * 3
msize[msize < 0] = 0.
ergoPlot.plotEigPowerRec(angFreq, eigValGen, powerSample, powerRec,
                         markersize=msize, condition=eigenCondition,
                         xlabel=realLabel, ylabel=imagLabel, zlabel=powerLabel,
                         xlim=xlimEig, ylim=ylimEig, zlim=zlimEig,
                         xticks=xticks, yticks=yticks, zticks=zticks)
plt.savefig('%s/spectrum/reconstruction/%sRec_chunk%d_nev%d%s.%s'\
            % (cfg.general.plotDir, powerName, int(cfg.stat.chunkWidth),
               cfg.spectrum.nev, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

