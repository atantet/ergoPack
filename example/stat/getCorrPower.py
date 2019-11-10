import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
from ergoPack import ergoStat, ergoPlot

configFile = '../cfg/Chekroun2019.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

L = cfg.simulation.LCut + cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt)
caseName = cfg.model.caseName
# delayName = ""
# if hasattr(cfg.model, 'delaysDays'):
#     for d in np.arange(len(cfg.model.delaysDays)):
#         delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

# if (hasattr(cfg.model, 'rho') & hasattr(cfg.model, 'sigma') & hasattr(cfg.model, 'beta')):
#     caseName = "%s_rho%d_sigma%d_beta%d" \
#                % (caseName, (int) (cfg.model.rho * 1000),
#                   (int) (cfg.model.sigma * 1000), (int) (cfg.model.beta * 1000))
# srcPostfix = "_%s%s_L%d_spinup%d_dt%d_samp%d" \
#              % (caseName, delayName, L, cfg.simulation.spinup,
#                 -np.round(np.log10(cfg.simulation.dt)), printStepNum)
srcPostfix = (
    "_{}_mu{:04d}_alpha{:04d}_gamma{:04d}_delta{:04d}_beta{:04d}_eps{:04d}_sep{:04d}"
    "_L{:d}_spinup{:d}_dt{:d}_samp{:d}".format(
        caseName, int(cfg.model.mu * 10000 + 0.1),
        int(cfg.model.alpha * 10000 + 0.1), int(cfg.model.gamma * 10000 + 0.1),
        int(cfg.model.delta * 10000 + 0.1), int(cfg.model.beta * 10000 + 0.1),
        int(cfg.model.eps * 10000 + 0.1), int(cfg.model.sep * 10000 + 0.1),
        int(L + 0.1), int(cfg.simulation.spinup + 0.1),
        int(-np.round(np.log10(cfg.simulation.dt)) + 0.1), printStepNum))

sampFreq = 1. / cfg.simulation.printStep
lagMaxNum = int(np.round(cfg.stat.lagMax / cfg.simulation.printStep))
lags = np.arange(-cfg.stat.lagMax, cfg.stat.lagMax + 0.999 *
                 cfg.simulation.printStep, cfg.simulation.printStep)
corrName = 'C{:d}{:d}'.format(cfg.stat.idxf, cfg.stat.idxg)
powerName = 'S{:d}{:d}'.format(cfg.stat.idxf, cfg.stat.idxg)

lagMaxSample = int(cfg.stat.lagMax * sampFreq + 0.1)
lags = np.arange(-cfg.stat.lagMax, cfg.stat.lagMax + 0.999 / sampFreq,
                 1. / sampFreq)
nLags = lags.shape[0]

nTraj = cfg.sprinkle.nTraj
dstPostfix = "{}_nTraj{:d}".format(srcPostfix, nTraj)
corrSample = np.zeros((nLags,))
for traj in np.arange(nTraj):
    print('for traj {:d}'.format(traj))

    # Read time series
    simFile = '{}/simulation/sim{}_traj{:d}.{}'.format(
        cfg.general.resDir, srcPostfix, traj, cfg.general.fileFormat)
    print('Reading time series from ' + simFile)
    if cfg.general.fileFormat == 'bin':
        X = np.fromfile(simFile, dtype=float,
                        count=int(np.round(cfg.model.dim * cfg.simulation.LCut
                                           / cfg.simulation.printStep)))
    else:
        X = np.loadtxt(simFile, dtype=float)
    X = X.reshape(-1, cfg.model.dim)

    # Read datasets
    observable1 = X[:, cfg.stat.idxf]
    observable2 = X[:, cfg.stat.idxg]
    nt = observable1.shape[0]
    ntWindow = int(cfg.stat.chunkWidth * sampFreq)

    # Get corrSample averaged over trajectories (should add weights based on length)
    # (do not normalize here, because we summup the trajectories)
    print('Computing correlation function')
    corrSample += ergoStat.ccf(observable1, observable2,
                               lagMax=cfg.stat.lagMax,
                               sampFreq=sampFreq, norm=False)

    # Get common frequencies
    if traj == 0:
        nChunks = int(nt / (cfg.stat.chunkWidth * sampFreq))
        freq = ergoStat.getFreqPow2(cfg.stat.chunkWidth,
                                    sampFreq=sampFreq)
        nfft = freq.shape[0]
        powerSample = np.zeros((nfft,))
        powerSampleSTD = np.zeros((nfft,))

    # Get powerSample averaged over trajectories
    # (should add weights based on length)
    print('Computing periodogram')
    (freq, powerSampleTraj, powerSampleSTDTraj) \
        = ergoStat.getPerio(observable1, observable2,
                            freq=freq, sampFreq=sampFreq,
                            chunkWidth=cfg.stat.chunkWidth, norm=False)
    powerSample += powerSampleTraj
    powerSampleSTD += powerSampleSTDTraj**2 * nChunks

corrSample /= nTraj
powerSample /= nTraj
powerSampleSTD = np.sqrt(powerSampleSTD / (nTraj * nChunks))
if cfg.stat.norm:
    cov = corrSample[(lags.shape[0] - 1) // 2]
    corrSample /= cov
    powerSample /= cov
    powerSampleSTD /= cov

# Save results
np.savetxt(os.path.join(
    cfg.general.resDir, 'correlation', 'corrSample{}_lagMax{:d}yr.txt'.format(
        dstPostfix, int(cfg.stat.lagMax * 1e4 + 0.1))), corrSample)
np.savetxt(os.path.join(
    cfg.general.resDir, 'correlation', 'lags{}_lagMax{:d}yr.txt'.format(
        dstPostfix, int(cfg.stat.lagMax * 1e4 + 0.1))), lags)
np.savetxt(os.path.join(
    cfg.general.resDir, 'power', 'powerSample{}_chunk{:d}yr.txt'.format(
        dstPostfix, int(cfg.stat.chunkWidth + 0.1))), powerSample)
np.savetxt(os.path.join(
    cfg.general.resDir, 'power', 'powerSampleSTD{}_chunk{:d}yr.txt'.format(
        dstPostfix, int(cfg.stat.chunkWidth + 0.1))), powerSampleSTD)
np.savetxt(os.path.join(
    cfg.general.resDir, 'power', 'freq{}_chunk{:d}yr.txt'.format(
        dstPostfix, int(cfg.stat.chunkWidth + 0.1))), freq)

# Plot corrSample
print('Plotting correlation function...')
(fig, ax) = ergoPlot.plotCCF(corrSample, lags, absUnit='y',
                             plotPositive=True)
plt.savefig(os.path.join(
    cfg.general.plotDir, 'correlation', 'corrSample{}_lagMax{:d}yr.{}'.format(
        dstPostfix, int(cfg.stat.lagMax * 1e4 + 0.1), ergoPlot.figFormat)),
    dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot powerSample
print('Plotting periodogram...')
angFreq = freq * 2 * np.pi
(fig, ax) = ergoPlot.plotPerio(powerSample, perioSTD=powerSampleSTD,
                               freq=angFreq,  plotPositive=True,
                               absUnit='', yscale='log',
                               xlim=(0, cfg.stat.angFreqMax),
                               ylim=(cfg.stat.powerMin, cfg.stat.powerMax))
fig.savefig(os.path.join(
    cfg.general.plotDir, 'power', 'powerSample{}_chunk{:d}yr.{}'.format(
        dstPostfix, int(cfg.stat.chunkWidth + 0.1), ergoPlot.figFormat)),
    dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

plt.show(block=False)
