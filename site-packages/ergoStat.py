import numpy as np
import pylibconfig2


def ccf(ts1, ts2, lagMax=None, sampFreq=1.):
    """ Cross-correlation function"""
    ts1 = (ts1 - ts1.mean()) / ts1.std()
    ts2 = (ts2 - ts2.mean()) / ts2.std()
    nt = ts1.shape[0]
    if lagMax is None:
        lagMax = nt - 1
    lagMaxSample = int(lagMax * sampFreq)
    ccf = np.empty((lagMaxSample*2+1,))
    for k in np.arange(lagMaxSample):
        ccf[k] = (ts1[:-(lagMaxSample-k)] * ts2[lagMaxSample-k:]).mean()
    ccf[lagMax] = (ts1 * ts2).mean()
    for k in np.arange(lagMaxSample):
        ccf[2*lagMaxSample-k] = (ts2[:-(lagMaxSample-k)] \
                                 * ts1[lagMaxSample-k:]).mean()
    return ccf


def ccovf(ts1, ts2, lagMax=None):
    """ Cross-covariance function"""
    ts1 = ts1 - ts1.mean()
    ts2 = ts2 - ts2.mean()
    nt = ts1.shape[0]
    if lagMax is None:
        lagMax = nt - 1
    ccovf = np.empty((lagMax*2+1,))
    for k in np.arange(lagMax):
        ccovf[k] = (ts1[:-(lagMax-k)] * ts2[lagMax-k:]).mean()
    ccovf[lagMax] = (ts1 * ts2).mean()
    for k in np.arange(lagMax):
        ccovf[2*lagMax-k] = (ts2[:-(lagMax-k)] * ts1[lagMax-k:]).mean()
    return ccovf

def getPerio(ts1, ts2, freq=None, sampFreq=1., chunkWidth=None, norm=False, window=None):
    ''' Get the periodogram of ts using a taping window of length tape window'''
    nt = ts1.shape[0]

    # If no chunkWidth given then do not tape
    if chunkWidth is None:
        chunkWidthNum = nt
    else:
        chunkWidthNum = int(chunkWidth * sampFreq)
    if window is None:
        window = np.ones(chunkWidthNum)
        
    nChunks = int(nt / chunkWidthNum)

    # Get frequencies if not given
    if freq is None:
        freq = getFreqPow2(chunkWidth, sampFreq=sampFreq)
    nfft = freq.shape[0]

    # Remove mean [and normalize]
    ts1 -= ts1.mean(0)
    ts2 -= ts2.mean(0)

    # Get periodogram averages over nChunks windows
    perio = np.zeros((nfft,))
    perioSTD = np.zeros((nfft,))
    for tape in np.arange(nChunks):
        ts1Tape = ts1[tape*chunkWidthNum:(tape+1)*chunkWidthNum]
        ts1Windowed = ts1Tape * window
        ts2Tape = ts2[tape*chunkWidthNum:(tape+1)*chunkWidthNum]
        ts2Windowed = ts2Tape * window
        # Fourier transform and shift zero frequency to center
        fts1 = np.fft.fft(ts1Windowed, nfft, 0)
        fts1 = np.fft.fftshift(fts1)
        fts2 = np.fft.fft(ts2Windowed, nfft, 0)
        fts2 = np.fft.fftshift(fts2)
        # Get periodogram
        pt = (fts1 * np.conjugate(fts2)).real / chunkWidthNum / sampFreq
        perio +=  pt
        perioSTD += pt**2
    perio /= nChunks
    perioSTD = np.sqrt(perioSTD / nChunks)

    if norm:
        perio /= np.cov(ts1, ts2)[0, 1]
        perioSTD /= np.cov(ts1, ts2)[0, 1]

    return (freq, perio, perioSTD)

def getFreqPow2(L, sampFreq=1., center=True):
    ''' Get frequency vector with maximum span given by the closest power of 2 (fft) of the lenght of the times series nt'''
    nt = L * sampFreq
    # Get nearest larger power of 2
    if np.log2(nt) != int(np.log2(nt)):
        nfft = 2**(int(np.log2(nt)) + 1)
    else:
        nfft = nt

    # Get frequencies
    freq = np.fft.fftfreq(nfft, d=1./sampFreq)

    # Shift zero frequency to center
    if center:
        freq = np.fft.fftshift(freq)

    return freq
