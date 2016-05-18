import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoStat

configFile = sys.argv[1]
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

delayName = ""
if hasattr(cfg.model, 'delaysDays'):
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

sampFreq = 1. / cfg.simulation.printStep
lagMaxNum = int(np.round(cfg.stat.lagMax / cfg.simulation.printStep))
lags = np.arange(-cfg.stat.lagMax, cfg.stat.lagMax + 0.999 * cfg.simulation.printStep, cfg.simulation.printStep)
corrName = 'C%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
powerName = 'S%d%d' % (cfg.stat.idxf, cfg.stat.idxg)

# Read time series
simFile = '%s/simulation/sim%s.%s' % (cfg.general.resDir, srcPostfix, cfg.simulation.file_format)
print 'Reading time series from ' + simFile
if cfg.simulation.file_format == 'bin':
    X = np.fromfile(simFile, dtype=float,
                    count=int(np.round(cfg.model.dim * cfg.simulation.LCut\
                                       / cfg.simulation.printStep)))
else:
    X = np.loadtxt(simFile, dtype=float)
X = X.reshape(-1, cfg.model.dim)

# Read datasets
observable1 = X[:, cfg.stat.idxf]
observable2 = X[:, cfg.stat.idxg]
nt = observable1.shape[0]
ntWindow = int(cfg.stat.chunkWidth * sampFreq)
time = np.arange(0, nt, cfg.simulation.printStep)


# Get ccf averaged over seeds (should add weights based on length)
print 'Calculating correlation function...'
ccf = ergoStat.ccf(observable1, observable2, lagMax=cfg.stat.lagMax,
                   sampFreq=sampFreq)


# Get perio averaged over seeds (should add weights based on length)
print 'Calculating periodogram function...'
(freq, perio, perioSTD) \
    = ergoStat.getPerio(observable1, observable2,
                        sampFreq=sampFreq, chunkWidth=cfg.stat.chunkWidth,
                        norm=False)


# Save results
np.savetxt('%s/correlation/%s_lag%d%s.txt'\
           % (cfg.general.resDir, corrName, int(cfg.stat.lagMax),
              srcPostfix), ccf)
np.savetxt('%s/correlation/lags_lag%d%s.txt'\
           % (cfg.general.resDir, int(cfg.stat.lagMax),
              srcPostfix), lags)
np.savetxt('%s/power/%s_chunk%d%s.txt'\
           % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
              srcPostfix), perio)
np.savetxt('%s/power/%sSTD_chunk%d%s.txt' \
           % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
              srcPostfix), perioSTD)
np.savetxt('%s/power/freq_chunk%d%s.txt' \
           % (cfg.general.resDir, cfg.stat.chunkWidth,
              srcPostfix), freq)

