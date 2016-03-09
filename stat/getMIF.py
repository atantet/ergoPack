import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoStat

configFile = sys.argv[1]
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

lagStep = cfg.simulation.printStep

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
lags = np.arange(0, cfg.stat.lagMax + 0.999, lagStep)

dimObs = len(cfg.observable.components)
obsName = ""
for d in np.arange(dimObs):
    obsName = "%s_c%d_e%d" % (obsName, cfg.observable.components[d],
                              cfg.observable.embeddingDays[d])

gridPostfix = ""
N = 1
for d in np.arange(dimObs):
    N *= cfg.grid.nx[d]
    if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.grid.nSTDLow[d], cfg.grid.nSTDHigh[d])
    else:
        gridPostfix = "%s_n%dminmax" % (gridPostfix, cfg.grid.nx[d])
gridPostfix = "%s%s%s" % (srcPostfix, obsName, gridPostfix)

# Read membership vector
gridMemFileName = "%s/transfer/gridMem/gridMem%s.txt" \
                  % (cfg.general.resDir, gridPostfix)
print 'Reading membership vector in ', gridMemFileName
gridMemVectAll = np.loadtxt(gridMemFileName, dtype=int)

gridMemVect= gridMemVectAll[:np.round(gridMemVectAll.shape[0])]

# Get mutual information function
print 'Getting mutual information function...'
mif = ergoStat.getMIFFromGridMemVect(gridMemVect, N, lagMax=cfg.stat.lagMax,
                                     sampFreq=sampFreq, step=lagStep, verbose=True)

# Save mutual information function
print 'Saving mutual information function...'
np.savetxt('%s/mutual_information/mif_lag%d%s.txt'\
           % (cfg.general.resDir, int(cfg.stat.lagMax),
              gridPostfix), mif)
np.savetxt('%s/mutual_information/lags_lag%d%s.txt'\
           % (cfg.general.resDir, int(cfg.stat.lagMax),
              gridPostfix), lags)



