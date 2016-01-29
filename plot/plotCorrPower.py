import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ergoPlot

configFile = sys.argv[1]
ergoPlot.readConfig(configFile)

sampFreq = 1. / ergoPlot.printStep
lagMaxNum = int(np.round(ergoPlot.lagMax / ergoPlot.printStep))
lags = np.arange(-ergoPlot.lagMax, ergoPlot.lagMax + 0.999 * ergoPlot.printStep,
                 ergoPlot.printStep)
corrName = 'C%d%d' % (ergoPlot.component1, ergoPlot.component2)
corrLabel = r'$C_{x_%d, x_%d}(t)$' % (ergoPlot.component1 + 1,
                                      ergoPlot.component2 + 1)
powerName = 'S%d%d' % (ergoPlot.component1, ergoPlot.component2)
powerLabel = r'$S_{x_%d, x_%d}(\omega)$' % (ergoPlot.component1 + 1,
                                            ergoPlot.component2 + 1)

# Read ccf
print 'Reading correlation function and periodogram...'
ccf = np.loadtxt('%s/correlation/%s_lag%d%s.txt'\
                 % (ergoPlot.resDir, corrName, int(ergoPlot.lagMax),
                    ergoPlot.srcPostfix))
lags = np.loadtxt('%s/correlation/lags_lag%d%s.txt'\
                  % (ergoPlot.resDir, int(ergoPlot.lagMax),
                     ergoPlot.srcPostfix))
perio = np.loadtxt('%s/power/%s_chunk%d%s.txt'\
                   % (ergoPlot.resDir, powerName, int(ergoPlot.chunkWidth),
                      ergoPlot.srcPostfix))
perioSTD = np.loadtxt('%s/power/%sSTD_chunk%d%s.txt' \
                      % (ergoPlot.resDir, powerName, int(ergoPlot.chunkWidth),
                         ergoPlot.srcPostfix))
freq = np.loadtxt('%s/power/freq_chunk%d%s.txt' \
                  % (ergoPlot.resDir, ergoPlot.chunkWidth,
                     ergoPlot.srcPostfix))
        
# Plot CCF
print 'Plotting correlation function...'
(fig, ax) = ergoPlot.plotCCF(ccf, lags, ylabel=corrLabel, plotPositive=True)
plt.savefig('%s/plot/correlation/%s_lag%d%s.%s'\
           % (ergoPlot.resDir, corrName, int(ergoPlot.lagMax),
              ergoPlot.srcPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot perio
print 'Plotting periodogram...'
angFreq = freq * 2 * np.pi
(fig, ax) = ergoPlot.plotPerio(perio, perioSTD=perioSTD, freq=angFreq,
                               ylabel=powerLabel, plotPositive=True,
                               absUnit='', yscale='log',
                               xlim=(0, ergoPlot.angFreqMax),
                               ylim=(ergoPlot.powerMin, ergoPlot.powerMax))
plt.savefig('%s/plot/power/%s_chunk%d%s.%s'\
            % (ergoPlot.resDir, powerName, int(ergoPlot.chunkWidth),
               ergoPlot.srcPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

