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
corrName = 'C%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
corrLabel = r'$C_{x_%d, x_%d}(t)$' % (cfg.stat.idxf + 1,
                                      cfg.stat.idxg + 1)
powerName = 'S%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
powerLabel = r'$S_{x_%d, x_%d}(\omega)$' % (cfg.stat.idxf + 1,
                                            cfg.stat.idxg + 1)

# Read ccf
print 'Reading correlation function and periodogram...'
ccf = np.loadtxt('%s/correlation/%s_lag%d%s.txt'\
                 % (cfg.general.resDir, corrName, int(cfg.stat.lagMax),
                    srcPostfix))
lags = np.loadtxt('%s/correlation/lags_lag%d%s.txt'\
                  % (cfg.general.resDir, int(cfg.stat.lagMax),
                     srcPostfix))
perio = np.loadtxt('%s/power/%s_chunk%d%s.txt'\
                   % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
                      srcPostfix))
perioSTD = np.loadtxt('%s/power/%sSTD_chunk%d%s.txt' \
                      % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
                         srcPostfix))
freq = np.loadtxt('%s/power/freq_chunk%d%s.txt' \
                  % (cfg.general.resDir, cfg.stat.chunkWidth,
                     srcPostfix))
        
# Plot CCF
print 'Plotting correlation function...'
(fig, ax) = ergoPlot.plotCCF(ccf, lags, ylabel=corrLabel, plotPositive=True)
plt.savefig('%s/plot/correlation/%s_lag%d%s.%s'\
           % (cfg.general.resDir, corrName, int(cfg.stat.lagMax),
              srcPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot perio
print 'Plotting periodogram...'
angFreq = freq * 2 * np.pi
(fig, ax) = ergoPlot.plotPerio(perio, perioSTD=perioSTD, freq=angFreq,
                               ylabel=powerLabel, plotPositive=True,
                               absUnit='', yscale='log',
                               xlim=(0, cfg.stat.angFreqMax),
                               ylim=(cfg.stat.powerMin, cfg.stat.powerMax))
plt.savefig('%s/plot/power/%s_chunk%d%s.%s'\
            % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
               srcPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

