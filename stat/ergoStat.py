import numpy as np
import pylibconfig2

def readConfig(configFile):
    "Read configuration file or plotSpectrum.py using libconfig for python"
    global gridFile, dim, dimObs, gridPostfix, specDir, plotDir, nev, resDir, lagMax, chunkWidth
    global file_format, printStep, component1, component2, srcPostfix
    global angFreqMax, powerMin, powerMax
    
    cfg = pylibconfig2.Config()
    cfg.read_file(configFile)

    if hasattr(cfg, 'general'):
        resDir = cfg.lookup("general.resDir")
        specDir = '%s/spectrum/' % resDir
        plotDir = '%s/plot/' % resDir
        
    delayName = ""
    if hasattr(cfg, 'model'):
        caseName = cfg.lookup("model.caseName")
        dim = cfg.lookup("model.dim")
        if hasattr(cfg.model, 'delaysDays'):
            delaysDays = np.array(cfg.lookup("model.delaysDays"))
            for d in np.arange(delaysDays.shape[0]):
                delayName = "%s_d%d" % (delayName, delaysDays[d])

    if hasattr(cfg, 'simulation'):
        LCut = cfg.lookup("simulation.LCut")
        dt = cfg.lookup("simulation.dt")
        spinup = cfg.lookup("simulation.spinup")
        printStep = cfg.lookup("simulation.printStep")
        L = LCut + spinup
        printStepNum = int(printStep / dt)
        file_format = cfg.lookup("simulation.file_format")
        srcPostfix = "_%s%s_L%d_spinup%d_dt%d_samp%d" \
                     % (caseName, delayName, L, spinup, -np.round(np.log10(dt)), printStepNum)

    obsName = ""
    if hasattr(cfg, 'observable'):
        components = np.array(cfg.lookup("observable.components"))
        embeddingDays = np.array(cfg.lookup("observable.embeddingDays"))
        embedding = (embeddingDays / 365 / printStep).astype(int)
        dimObs = components.shape[0]
        for d in np.arange(dimObs):
	    obsName = "%s_c%d_e%d" % (obsName, components[d], embeddingDays[d])

    gridCFG = ""
    if hasattr(cfg, 'grid'):
        nx = np.array(cfg.lookup("grid.nx"))
        nSTDLow = np.array(cfg.lookup("grid.nSTDLow"))
        nSTDHigh = np.array(cfg.lookup("grid.nSTDHigh"))
        N = np.prod(nx)
        for d in np.arange(dimObs):
            gridCFG = "%s_n%dl%dh%d" % (gridCFG, nx[d], nSTDLow[d], nSTDHigh[d])
        gridPostfix = "%s%s%s" % (srcPostfix, obsName, gridCFG)
        gridFile = '%s/grid/grid%s.txt' % (resDir, gridPostfix)

    if hasattr(cfg, 'transfer'):
        tauRng = np.array(cfg.lookup("transfer.tauRng"))

    if hasattr(cfg, 'spectrum'):
        nev = cfg.lookup("spectrum.nev");

    if hasattr(cfg, 'stat'):
        component1 = cfg.lookup('stat.component1')
        component2 = cfg.lookup('stat.component2')
        lagMax = cfg.lookup('stat.lagMax')
        chunkWidth = cfg.lookup('stat.chunkWidth')
        angFreqMax = cfg.lookup('stat.angFreqMax')
        powerMin = cfg.lookup('stat.powerMin')
        powerMax = cfg.lookup('stat.powerMax')
        

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
        
    nTapes = int(nt / chunkWidthNum)

    # Get frequencies if not given
    if freq is None:
        freq = getFreqPow2(chunkWidth, sampFreq=sampFreq)
        nfft = freq.shape[0]

    # Remove mean [and normalize]
    ts1 -= ts1.mean(0)
    ts2 -= ts2.mean(0)

    # Get periodogram averages over nTapes windows
    perio = np.zeros((nfft,))
    perioSTD = np.zeros((nfft,))
    for tape in np.arange(nTapes):
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
    perio /= nTapes
    perioSTD = np.sqrt(perioSTD / nTapes)

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
