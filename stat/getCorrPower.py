import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ergoStat

configFile = sys.argv[1]
ergoStat.readConfig(configFile)

sampFreq = 1. / ergoStat.printStep
lagMaxNum = int(np.round(ergoStat.lagMax / ergoStat.printStep))
lags = np.arange(-ergoStat.lagMax, ergoStat.lagMax + 0.999 * ergoStat.printStep, ergoStat.printStep)
corrName = 'C%d%d' % (ergoStat.component1, ergoStat.component2)
powerName = 'S%d%d' % (ergoStat.component1, ergoStat.component2)

# Read time series
simFile = '%s/simulation/sim%s.%s' % (ergoStat.resDir, ergoStat.srcPostfix, ergoStat.file_format)
print 'Reading time series from ' + simFile
if ergoStat.file_format == 'bin':
    X = np.fromfile(simFile, dtype=float)
else:
    X = np.loadtxt(simFile, dtype=float)
X = X.reshape(np.prod(X.shape) / ergoStat.dim, ergoStat.dim)

# Read datasets
observable1 = X[:, ergoStat.component1]
observable2 = X[:, ergoStat.component2]
nt = X.shape[0]
ntWindow = int(ergoStat.chunkWidth * sampFreq)
time = np.arange(0, nt, ergoStat.printStep)


# Get ccf averaged over seeds (should add weights based on length)
print 'Calculating correlation function...'
ccf = ergoStat.ccf(observable1, observable2, lagMax=ergoStat.lagMax,
                   sampFreq=sampFreq)


# Get perio averaged over seeds (should add weights based on length)
print 'Calculating periodogram function...'
nTapes = int(nt / ntWindow)
freq = ergoStat.getFreqPow2(ntWindow, sampFreq=sampFreq)
nfft = freq.shape[0]
perio = np.zeros((nfft,))
perioSTD = np.zeros((nfft,))
                
(freq, perio, perioSTD) \
    = ergoStat.getPerio(observable1, observable2,
                        freq=freq, sampFreq=sampFreq,
                        chunkWidth=ergoStat.chunkWidth)
perio = perio
perioSTD = np.sqrt(perioSTD**2)


# Save results
np.savetxt('%s/correlation/%s_lag%d%s.txt'\
           % (ergoStat.resDir, corrName, int(ergoStat.lagMax),
              ergoStat.srcPostfix), ccf)
np.savetxt('%s/correlation/lags_lag%d%s.txt'\
           % (ergoStat.resDir, int(ergoStat.lagMax),
              ergoStat.srcPostfix), lags)
np.savetxt('%s/power/%s_chunk%d%s.txt'\
           % (ergoStat.resDir, powerName, int(ergoStat.chunkWidth),
              ergoStat.srcPostfix), perio)
np.savetxt('%s/power/%sSTD_chunk%d%s.txt' \
           % (ergoStat.resDir, powerName, int(ergoStat.chunkWidth),
              ergoStat.srcPostfix), perioSTD)
np.savetxt('%s/power/freq_chunk%d%s.txt' \
           % (ergoStat.resDir, ergoStat.chunkWidth,
              ergoStat.srcPostfix), freq)

