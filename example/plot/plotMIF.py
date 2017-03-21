import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

configFile = sys.argv[1]
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

sampFreq = 1. / cfg.simulation.printStep
lagMaxNum = int(np.round(cfg.stat.lagMax / cfg.simulation.printStep))
lags = np.arange(-cfg.stat.lagMax, cfg.stat.lagMax + 0.999 * cfg.simulation.printStep,
                 cfg.simulation.printStep)
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

mifName = 'mif'
mifLabel = r'$MI(t)$'

# Read ccf
print 'Reading mutual information function...'
mif = np.loadtxt('%s/mutual_information/mif_lag%d%s.txt'\
                 % (cfg.general.resDir, int(cfg.stat.lagMax),
                    gridPostfix))
lags = np.loadtxt('%s/mutual_information/lags_lag%d%s.txt'\
                  % (cfg.general.resDir, int(cfg.stat.lagMax),
                     gridPostfix))
        
# Plot MIF
print 'Plotting mutual information function...'
(fig, ax) = ergoPlot.plotCCF(mif, lags, ylabel=mifLabel, plotPositive=False,
                             ylim=[0., mif.max()])
plt.savefig('%s/plot/mutual_information/%s_lag%d%s.%s'\
           % (cfg.general.resDir, mifName, int(cfg.stat.lagMax),
              gridPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
