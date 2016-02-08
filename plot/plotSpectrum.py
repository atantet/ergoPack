import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import ergoPlot

#configFile = '../cfg/OU2d.cfg'
#ev_xlabel = r'$x_1$'
#ev_ylabel = r'$x_2$'
configFile = '../cfg/Battisti1989.cfg'
#configFile = '../cfg/Suarez1988.cfg'
ev_xlabel = r'$y(t)$'
ev_ylabel = r'$y(t - \tau)$'

tau = 0.05
#tau = 0.15

nevPlot = 5
#plotBackward = False
plotBackward = True
plotImag = False
#plotImag = True
nComponents = 15

# Read configuration
ergoPlot.readConfig(configFile)
xminEigVal = -ergoPlot.rateMax
yminEigVal = -ergoPlot.angFreqMax

# Read grid
(X, Y) = ergoPlot.readGrid(ergoPlot.gridFile, ergoPlot.dimObs)
coord = (X.flatten(), Y.flatten())

# Define file names
postfix = "%s_tau%03d" % (ergoPlot.gridPostfix, tau * 1000)
EigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.txt' \
                    % (ergoPlot.specDir, ergoPlot.nev, postfix)
EigVecForwardFile = '%s/eigvec/eigvecForward_nev%d%s.txt' \
                    % (ergoPlot.specDir, ergoPlot.nev, postfix)
EigValBackwardFile = '%s/eigval/eigvalBackward_nev%d%s.txt' \
                    % (ergoPlot.specDir, ergoPlot.nev, postfix)
EigVecBackwardFile = '%s/eigvec/eigvecBackward_nev%d%s.txt' \
                    % (ergoPlot.specDir, ergoPlot.nev, postfix)

# Read stationary distribution
statDist = np.loadtxt('%s/transfer/initDist/initDist%s.txt' % (ergoPlot.resDir, postfix))

# Read transfer operator spectrum from file and create a bi-orthonormal basis
# of eigenvectors and adjoint eigenvectors:
print 'Readig spectrum...'
(eigValForward, eigVecForward, eigValBackward, eigVecBackward) \
    = ergoPlot.readSpectrum(ergoPlot.nev, EigValForwardFile, EigVecForwardFile,
                            EigValBackwardFile, EigVecBackwardFile, statDist,
                            makeBiorthonormal=False)

print 'Getting conditionning of eigenvectors...'
eigenCondition = ergoPlot.getEigenCondition(eigVecForward, eigVecBackward, statDist)

# Get generator eigenvalues
eigValGen = (np.log(np.abs(eigValForward)) + np.angle(eigValForward)*1j) / tau


# Plot eigenvectors of transfer operator
alpha = 0.01
for ev in np.arange(nevPlot):
    print 'Plotting real part of eigenvector %d...' % (ev + 1,)
    ergoPlot.plot2D(X, Y, eigVecForward[:, ev].real, ev_xlabel, ev_ylabel, alpha)
    dstFile = '%s/spectrum/eigvec/eigvecForwardReal_nev%d_ev%03d%s.%s' \
              % (ergoPlot.plotDir, ergoPlot.nev, ev + 1, postfix, ergoPlot.figFormat)
    plt.savefig(dstFile, bbox_inches='tight', dpi=ergoPlot.dpi)
    
    if plotImag & (eigValForward[ev].imag != 0):
        print 'Plotting imaginary  part of eigenvector %d...' % (ev + 1,)
        ergoPlot.plot2D(X, Y, eigVecForward[:, ev].imag, ev_xlabel, ev_ylabel, alpha)
        dstFile = '%s/spectrum/eigvec/eigvecForwardImag_nev%d_ev%03d%s.%s' \
                  % (ergoPlot.plotDir, ergoPlot.nev, ev + 1, postfix, ergoPlot.figFormat)
        plt.savefig(dstFile, bbox_inches='tight', dpi=ergoPlot.dpi)
    
    # Plot eigenvectors of adjoint operator
    if plotBackward:
        print 'Plotting real part of adjoint eigenvector %d...' % (ev + 1,)
        ergoPlot.plot2D(X, Y, eigVecBackward[:, ev].real, ev_xlabel, ev_ylabel, alpha)
        dstFile = '%s/spectrum/eigvec/eigvecBackwardReal_nev%d_ev%03d%s.%s' \
                  % (ergoPlot.plotDir, ergoPlot.nev, ev + 1, postfix, ergoPlot.figFormat)
        plt.savefig(dstFile, bbox_inches='tight', dpi=ergoPlot.dpi)
        
        if plotImag & (eigValForward[ev].imag != 0):
            print 'Plotting imaginary  part of adjoint eigenvector %d...' % (ev + 1,)
            ergoPlot.plot2D(X, Y, eigVecBackward[:, ev].imag, ev_xlabel, ev_ylabel, alpha)
            dstFile = '%s/spectrum/eigvec/eigvecBackwardImag_nev%d_ev%03d%s.%s' \
                      % (ergoPlot.plotDir, ergoPlot.nev, ev + 1, postfix, ergoPlot.figFormat)
            plt.savefig(dstFile, bbox_inches='tight', dpi=ergoPlot.dpi)

# Define observables
corrName = 'C%d%d' % (ergoPlot.component1, ergoPlot.component2)
powerName = 'S%d%d' % (ergoPlot.component1, ergoPlot.component2)
f = coord[ergoPlot.component1]
g = coord[ergoPlot.component2]
corrLabel = r'$C_{x_%d, x_%d}(t)$' % (ergoPlot.component1 + 1,
                                      ergoPlot.component2 + 1)
powerLabel = r'$S_{x_%d, x_%d}(\omega)$' % (ergoPlot.component1 + 1,
                                            ergoPlot.component2 + 1)
realLabel = r'$\Re(\bar{\lambda}_k)$'
imagLabel = r'$\Im(\bar{\lambda}_k)$'

# Read ccf
print 'Reading correlation function and periodogram...'
corrSample = np.loadtxt('%s/correlation/%s_lag%d%s.txt'\
                        % (ergoPlot.resDir, corrName, int(ergoPlot.lagMax),
                           ergoPlot.srcPostfix))
lags = np.loadtxt('%s/correlation/lags_lag%d%s.txt'\
                  % (ergoPlot.resDir, int(ergoPlot.lagMax),
                     ergoPlot.srcPostfix))
powerSample = np.loadtxt('%s/power/%s_chunk%d%s.txt'\
                         % (ergoPlot.resDir, powerName, int(ergoPlot.chunkWidth),
                            ergoPlot.srcPostfix))
powerSampleSTD = np.loadtxt('%s/power/%sSTD_chunk%d%s.txt' \
                            % (ergoPlot.resDir, powerName, int(ergoPlot.chunkWidth),
                               ergoPlot.srcPostfix))
freq = np.loadtxt('%s/power/freq_chunk%d%s.txt' \
                  % (ergoPlot.resDir, ergoPlot.chunkWidth,
                     ergoPlot.srcPostfix))

# Convert to angular frequencies and normalize by covariance
angFreq = freq * 2*np.pi
cfg0 = ((f - (f * statDist).sum()) * statDist * (g - (g * statDist).sum())).sum()
powerSample /= 2 * np.pi * cfg0
powerSampleSTD /= 2 * np.pi * cfg0

# Get error bars
powerSampleDown = powerSample - powerSampleSTD / 2
powerSampleUp = powerSample + powerSampleSTD / 2

# Reconstruct correlation and power spectrum
# Get normalized weights
weights = ergoPlot.getSpectralWeights(f, g, eigVecForward, eigVecBackward,
                                      statDist, nComponents, skipMean=True)
(corrRec, compCorrRec) = ergoPlot.spectralRecCorrelation(lags, f, g, eigValGen, weights,
                                                         statDist, nComponents, skipMean=True)
(powerRec, compPowerRec) = ergoPlot.spectralRecPower(angFreq, f, g, eigValGen, weights,
                                                     statDist, nComponents)

# Plot correlation reconstruction
ergoPlot.plotRecCorrelation(lags, corrSample, corrRec, plotPositive=True,
                            ylabel=corrLabel)
plt.savefig('%s/spectrum/reconstruction/%sRec_lag%d_nev%d%s.%s'\
            % (ergoPlot.plotDir, corrName, int(ergoPlot.lagMax),
               ergoPlot.nev, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# PLot spectrum, powerSampledogram and spectral reconstruction
msizeWeight = np.log10(weights.real)
msizeWeight = (msizeWeight - msizeWeight[~np.isnan(msizeWeight)].min()) * 10
#msizeWeight = (msizeWeight - msizeWeight[~np.isnan(msizeWeight)].min()) * 3
msizeWeight[np.isnan(msizeWeight)] = 0.
ergoPlot.plotEigPowerRec(angFreq, eigValGen, msizeWeight, powerSample, powerSampleSTD,
                         powerRec, xlabel=realLabel, ylabel=imagLabel, zlabel=powerLabel,
                         xlim=[xminEigVal, -xminEigVal/100],
                         ylim=[yminEigVal, -yminEigVal],
                         zlim=[ergoPlot.powerMin, ergoPlot.powerMax])
plt.savefig('%s/spectrum/reconstruction/%sRec_chunk%d_nev%d%s.%s'\
            % (ergoPlot.plotDir, powerName, int(ergoPlot.chunkWidth),
               ergoPlot.nev, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

