import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

ergoPlot.dpi = 2000

#configFile = '../cfg/OU2d.cfg'
#compName1 = 'x_1'
#compName2 = 'x_2'
#configFile = '../cfg/Battisti1989.cfg'
configFile = '../cfg/Suarez1988.cfg'
compName1 = r'y'
compName2 = r'y'
#configFile = '../cfg/Lorenz63.cfg'
#compName1 = 'x'
#compName2 = 'y'
#compName3 = 'z'

cfg = pylibconfig2.Config()
cfg.read_file(configFile)

# Transition lag
if (hasattr(cfg.stat, 'tauPlot')):
    tau = cfg.stat.tauPlot
else:
    tau = cfg.transfer.tauRng[0]
unfold = False
tauUnfold = None
if (hasattr(cfg.stat, 'unfold')) & (hasattr(cfg.stat, 'tauUnfold')):
    if cfg.stat.unfold:
        unfold = True
        tauUnfold = cfg.stat.tauUnfold

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
xlabelCorr = r'$t$'

nevPlot = 6
xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
#plotBackward = False
plotBackward = True
plotImag = False
#plotImag = True
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
zlimEig = [cfg.stat.powerMin, cfg.stat.powerMax]
xticks = None
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))
zticks = np.logspace(np.log10(zlimEig[0]), np.log10(zlimEig[1]),
                    int(np.round(np.log10(zlimEig[1]/zlimEig[0]) + 1)))
zticks = np.logspace(np.log10(zlimEig[0]), np.log10(zlimEig[1]),
                     int(np.round(np.log10(zlimEig[1]/zlimEig[0])/2 + 1)))
alpha = 0.01
nev = cfg.spectrum.nev

# Read grid
coord = ergoPlot.readGrid(gridFile, dimObs)
if dimObs == 1:
    X = coord[0]
elif dimObs == 2:
    X, Y = np.meshgrid(coord[0], coord[1])
    coord = (X.flatten(), Y.flatten())
elif dimObs == 3:
    X, Y, Z = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
    coord = (X.T.flatten(), Y.T.flatten(), Z.T.flatten())

# Define file names
postfix = "%s_tau%03d" % (gridPostfix, tau * 1000)
eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.txt' \
                    % (cfg.general.specDir, nev, postfix)
eigVecForwardFile = '%s/eigvec/eigvecForward_nev%d%s.txt' \
                    % (cfg.general.specDir, nev, postfix)
eigValBackwardFile = '%s/eigval/eigvalBackward_nev%d%s.txt' \
                    % (cfg.general.specDir, nev, postfix)
eigVecBackwardFile = '%s/eigvec/eigvecBackward_nev%d%s.txt' \
                    % (cfg.general.specDir, nev, postfix)
statDistFile = '%s/transfer/initDist/initDist%s.txt' % (cfg.general.resDir, gridPostfix)
if unfold:
    postfixUnfold = "%s_tau%03d" % (gridPostfix, tauUnfold * 1000)
    eigValForwardFileUnfold = '%s/eigval/eigvalForward_nev%d%s.txt' \
                              % (cfg.general.specDir, nev, postfixUnfold)
    eigVecForwardFileUnfold = '%s/eigvec/eigvecForward_nev%d%s.txt' \
                              % (cfg.general.specDir, nev, postfixUnfold)
    

# # Read transfer operator spectrum from file and create a bi-orthonormal basis
# # of eigenvectors and backward eigenvectors:
# print 'Readig spectrum for tau = %.3f...' % tau
# (eigValForward, eigValBackward, statDist, eigVecForward, eigVecBackward) \
#     = ergoPlot.readSpectrum(eigValForwardFile, eigValBackwardFile, statDistFile,
#                             eigVecForwardFile, eigVecBackwardFile,
#                             makeBiorthonormal=~cfg.spectrum.makeBiorthonormal)

# print 'Getting conditionning of eigenvectors...'
# eigenCondition = ergoPlot.getEigenCondition(eigVecForward, eigVecBackward, statDist)

# tauMax = np.max([tau, tauUnfold])
# eigValGenOrig = ergoPlot.eig2Generator(eigValForward, tau)
# eigValGenShift = np.log(np.abs(eigValForward)) / tau \
#                  + 1j*(np.mod(np.angle(eigValForward) / tau+np.pi/tauMax,
#                               2*np.pi/tauMax)-np.pi/tauMax)

# Read unfolding eigenvalues
#eigValForwardUnfold = None
st = statDist.copy()
st2 = np.concatenate((st, st))
#st = np.ones(statDist.shape) / statDist.shape[0]
if unfold:
    # print 'Readig spectrum for tau = %.3f to unfold...' % tauUnfold
    # (eigValForwardUnfold, eigVecForwardUnfold) \
    #     = ergoPlot.readSpectrum(eigValForwardFileUnfold,
    #                             eigVecForwardFile=eigVecForwardFileUnfold)
    # eigVecForwardUnfold *= 1000
    
    # Sort thanks to the distance between the eigenvectors
    print 'Sorting...'
    isortUnfold = np.empty((nev,), dtype=int)
    eigVecForwardUnfold = eigVecForwardUnfold[:, np.argsort(-np.abs(eigValForwardUnfold))]
    for ev in np.arange(nev):
        eigVecDist = np.empty((nev,))
        vec = eigVecForward[:, ev].copy()
        B = np.matrix(np.empty((N*2, 1)))
        B[:N, 0] = np.matrix(vec.real).T
        B[N:, 0] = np.matrix(vec.imag).T
        Ba = np.array(B)[:, 0]
        for iev in np.arange(nev):
            # Get their distance to a phase shift
            vecUnfold = eigVecForwardUnfold[:, iev].copy()
            Z = np.matrix(np.empty((N*2, 2)))
            Z[:N, 0] = np.matrix(vecUnfold.real).T
            Z[N:, 0] = np.matrix(vecUnfold.imag).T
            Z[:N, 1] = -np.matrix(vecUnfold.imag).T
            Z[N:, 1] = np.matrix(vecUnfold.real).T
            Zd = np.matrix(np.empty((2, N*2)))
            Zd[0, :N] = np.matrix(vecUnfold.real * st)
            Zd[0, N:] = np.matrix(vecUnfold.imag * st)
            Zd[1, :N] = -np.matrix(vecUnfold.imag * st)
            Zd[1, :N] = np.matrix(vecUnfold.real * st)
            ZdZ = Zd * Z
            detZdZ = ZdZ[0,0]*ZdZ[1,1] - ZdZ[0,1]*ZdZ[1,0]
            if np.abs(detZdZ) < 1.e-10:
                A = np.matrix(np.zeros((2, 1)))
            else:
                ZdZm1 = np.matrix(np.empty((2, 2)))
                ZdZm1[0, 0] = ZdZ[1, 1]
                ZdZm1[0, 1] = -ZdZ[0, 1]
                ZdZm1[1, 0] = -ZdZ[1, 0]
                ZdZm1[1, 1] = ZdZ[0, 0]
                ZdZm1 /= detZdZ
                A = ZdZm1 * (Zd*B)
            ZAa = np.array(Z * A)[:, 0]
            Sres = np.sum((Ba - ZAa)**2 * st2)
            eigVecDist[iev] = Sres
        isortUnfold[ev] = np.argmin(eigVecDist)
#        print eigVecDist
        print '%03d <-> %03d (dist = %.12f)' % (ev, isortUnfold[ev], eigVecDist[isortUnfold[ev]])
        # vecUnfold = eigVecForwardUnfold[:, isortUnfold[ev]]
        # a = np.sum(vecUnfold * st * vecUnfold)**(-1) * np.sum(vecUnfold * st * vec)
        # ergoPlot.plot2D(X, Y, vec.real, ev_xlabel, ev_ylabel)
        # plt.title('ev.real = %d' % ev)
        # ergoPlot.plot2D(X, Y, (a*vecUnfold).real, ev_xlabel, ev_ylabel)
        # plt.title('ev.real unfold = %d' % isortUnfold[ev])
        # if np.abs(eigValForward[ev].imag) > 1.e-7:
        #     ergoPlot.plot2D(X, Y, vec.imag, ev_xlabel, ev_ylabel)
        #     plt.title('ev.imag = %d' % ev)
        #     ergoPlot.plot2D(X, Y, (a*vecUnfold).imag, ev_xlabel, ev_ylabel)
        #     plt.title('ev.imag unfold = %d' % isortUnfold[ev])
    eigValForwardUnfoldSort = eigValForwardUnfold[isortUnfold]

# Get generator eigenvalues
print 'Converting to generator eigenvalues...'
eigValGen = ergoPlot.eig2Generator(eigValForward, tau, eigValForwardUnfoldSort, tauUnfold)

# Plot eigenvectors of transfer operator
for ev in np.arange(nevPlot):
    print 'Plotting real part of eigenvector %d...' % (ev + 1,)
    #ergoPlot.plot2D(X, Y, eigVecForward[:, ev].real, ev_xlabel, ev_ylabel, alpha)
    ergoPlot.plot2D(X, Y, eigVecForward[:, ev].real, ev_xlabel, ev_ylabel, alpha)
    dstFile = '%s/spectrum/eigvec/eigvecForwardReal_nev%d_ev%03d%s.%s' \
              % (cfg.general.plotDir, nev, ev + 1, postfix, ergoPlot.figFormat)
    plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    
    if plotImag & (eigValForward[ev].imag != 0):
        print 'Plotting imaginary  part of eigenvector %d...' % (ev + 1,)
#        ergoPlot.plot2D(X, Y, eigVecForward[:, ev].imag, ev_xlabel, ev_ylabel, alpha)
        ergoPlot.plot2D(X, Y, eigVecForward[:, ev].imag, ev_xlabel, ev_ylabel, alpha)
        dstFile = '%s/spectrum/eigvec/eigvecForwardImag_nev%d_ev%03d%s.%s' \
                  % (cfg.general.plotDir, nev, ev + 1, postfix, ergoPlot.figFormat)
        plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    
    # Plot eigenvectors of backward operator
    if plotBackward:
        print 'Plotting real part of backward eigenvector %d...' % (ev + 1,)
        ergoPlot.plot2D(X, Y, eigVecBackward[:, ev].real, ev_xlabel, ev_ylabel, alpha)
        dstFile = '%s/spectrum/eigvec/eigvecBackwardReal_nev%d_ev%03d%s.%s' \
                  % (cfg.general.plotDir, nev, ev + 1, postfix, ergoPlot.figFormat)
        plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
        
        if plotImag & (eigValForward[ev].imag != 0):
            print 'Plotting imaginary  part of backward eigenvector %d...' % (ev + 1,)
            ergoPlot.plot2D(X, Y, eigVecBackward[:, ev].imag, ev_xlabel, ev_ylabel, alpha)
            dstFile = '%s/spectrum/eigvec/eigvecBackwardImag_nev%d_ev%03d%s.%s' \
                      % (cfg.general.plotDir, nev, ev + 1, postfix, ergoPlot.figFormat)
            plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)

            
# Define observables
corrName = 'C%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
powerName = 'S%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
f = coord[cfg.stat.idxf]
g = coord[cfg.stat.idxg]
# corrLabel = r'$C_{x_%d, x_%d}(t)$' % (cfg.stat.idxf + 1,
#                                       cfg.stat.idxg + 1)
# powerLabel = r'$S_{x_%d, x_%d}(\omega)$' % (cfg.stat.idxf + 1,
#                                             cfg.stat.idxg + 1)
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'

# Read ccf
print 'Reading correlation function and periodogram...'
srcPostfixStat = "_%s%s_L%d_spinup%d_dt%d_samp%d" \
                 % (caseName, delayName, L, cfg.simulation.spinup,
                    -np.round(np.log10(cfg.simulation.dt)), int(0.05 / cfg.simulation.dt))
corrSample = np.loadtxt('%s/correlation/%s_lag%d%s.txt'\
                        % (cfg.general.resDir, corrName,
                           int(cfg.stat.lagMax), srcPostfixStat))
lags = np.loadtxt('%s/correlation/lags_lag%d%s.txt'\
                  % (cfg.general.resDir, int(cfg.stat.lagMax),
                     srcPostfixStat))
powerSample = np.loadtxt('%s/power/%s_chunk%d%s.txt'\
                         % (cfg.general.resDir, powerName,
                            int(cfg.stat.chunkWidth), srcPostfixStat))
freq = np.loadtxt('%s/power/freq_chunk%d%s.txt' \
                  % (cfg.general.resDir, cfg.stat.chunkWidth,
                     srcPostfixStat))

# Convert to angular frequencies and normalize by covariance
angFreq = freq * 2*np.pi
cfg0 = ((f - (f * statDist).sum()) * statDist \
        * (g - (g * statDist).sum())).sum()
powerSample /= 2 * np.pi * cfg0

# Reconstruct correlation and power spectrum
# Get normalized weights
weights = ergoPlot.getSpectralWeights(f, g, eigVecForward, eigVecBackward,
                                      statDist, skipMean=True)
# Remove components with heigh condition number
weights[eigenCondition > cfg.stat.maxCondition] = 0.
condition = np.empty(eigenCondition.shape, dtype='S1')
condition[:] = 'k'
condition[eigenCondition > cfg.stat.maxCondition - 0.001] = 'w'
(corrRec, compCorrRec) = ergoPlot.spectralRecCorrelation(lags, f, g,
                                                         eigValGen, weights,
                                                         statDist,
                                                         skipMean=True,
                                                         norm=True)
(powerRec, compPowerRec) = ergoPlot.spectralRecPower(angFreq, f, g,
                                                     eigValGen, weights,
                                                     statDist, norm=True)

# Plot correlation reconstruction
ergoPlot.plotRecCorrelation(lags, corrSample, corrRec, plotPositive=True,
                            ylabel=corrLabel, xlabel=xlabelCorr)
plt.savefig('%s/spectrum/reconstruction/%sRec_lag%d_nev%d%s.%s'\
            % (cfg.general.plotDir, corrName, int(cfg.stat.lagMax),
               nev, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# PLot spectrum, powerSampledogram and spectral reconstruction
weights /= cfg0
msize = np.zeros((weights.shape[0]))
msize[weights.real > 0] = np.log10(weights[weights.real > 0].real)
msize[weights.real > 0] = (msize[weights.real > 0] + 8) * 10
# msize[weights.real > 0] = (msize[weights.real > 0] + 6) * 3
msize[msize < 0] = 0.
ergoPlot.plotEigPowerRec(angFreq, eigValGen, powerSample, powerRec,
                         markersize=msize, condition=condition,
                         xlabel=realLabel, ylabel=imagLabel,
                         zlabel=powerLabel,
                         xlim=xlimEig, ylim=ylimEig, zlim=zlimEig,
                         xticks=xticks, yticks=yticks, zticks=zticks)
plt.savefig('%s/spectrum/reconstruction/%sRec_chunk%d_nev%d%s.%s'\
            % (cfg.general.plotDir, powerName, int(cfg.stat.chunkWidth),
               nev, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

