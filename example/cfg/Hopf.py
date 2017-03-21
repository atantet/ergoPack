import os
import numpy as np
import pylibconfig2
import matplotlib.pyplot as plt
import ergoPlot

p = {}
readFile = np.fromfile
configFile = '../cfg/Hopf.cfg'

def getModelParam():
  cfg = pylibconfig2.Config()
  cfg.read_file(configFile)
  p['mu'] = cfg.model.mu
  p['beta'] = cfg.model.beta
  p['gamma'] = cfg.model.gamma
  p['eps'] = cfg.model.eps
  print 'mu = ', p["mu"]
  print 'beta = ', p["beta"]
  print 'gamma = ', p["gamma"]
  print 'eps = ', p["eps"]

  # Directories
  resDir = cfg.general.resDir
  plotDir = cfg.general.plotDir
  os.system("mkdir %s/continuation/ 2> /dev/null" % plotDir)
  os.system("mkdir %s/continuation/fp/ 2> /dev/null" % plotDir)
  os.system("mkdir %s/continuation/po/ 2> /dev/null" % plotDir)
  os.system("mkdir %s/continuation/phase/ 2> /dev/null" % plotDir)

  return cfg


def fieldHopf(X, p, dummy=None):
    (x, y) = X
    r2 = x**2 + y**2

    f = np.array([(p["mu"] - r2) * x - (p["gamma"] - p["beta"] * r2) * y,
		  (p["gamma"] - p["beta"] * r2) * x + (p["mu"] - r2) * y])
    
    return f

def JacobianHopf(X, p):
  (x, y) = X
  xy = x * y
  x2 = x**2
  y2 = y**2
  r2 = x2 + y2

  J = np.array([[p["mu"] - r2 - 2*x2 + 2*p["beta"] * xy,
                 -2*xy - (p["gamma"] - p["beta"] * r2) + 2*p["beta"] * y2],
                [p["gamma"] - p["beta"] * r2 - 2*p["beta"] * x2 - 2 * xy,
                 -2*p["beta"] * xy + (p["mu"] - r2) - 2*y2]])
  
  return J


def JacFieldHopf(dX, p, X):
    return np.dot(JacobianHopf(X, p), dX)


def plotFloquetVec(xt, p, FE, FVL, FVR, comps, scale=1, colors=None,
                   compLabels=None):
  (i, j) = comps
  po = xt[0]
  dim = po.shape[0]
  if colors is None:
    colors = ['r', 'g', 'b']
  if compLabels is None:
    compLabels = [r'$x$', r'$y$', r'$z$']
  # Plot (x, y) plane
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(xt[:, i], xt[:, j], '-k')
  ax.scatter(po[i], po[j], s=40, c='k')
  labels = ['0', '1', '2']
  for d in np.arange(dim):
    # Normalize only with respect to components
    vr = FVR[:, d].copy()
    vr /= np.sqrt(vr[i]**2 + vr[j]**2)
    ax.plot([po[i], po[i] + scale*vr[i]],
            [po[j], po[j] + scale*vr[j]], color=colors[d],
            linestyle='-', linewidth=2, label=r'$e^{%s}$' % labels[d])
  for d in np.arange(dim):
    vl = FVL[:, d].copy()
    vl /= np.sqrt(vl[i]**2 + vl[j]**2)
    ax.plot([po[i], po[i] + scale*vl[i]],
            [po[j], po[j] + scale*vl[j]], color=colors[d],
            linestyle='--', linewidth=2, label=r'$f^{%s}$' % labels[d])
  ax.legend(fontsize=ergoPlot.fs_latex, ncol=2, loc='lower center')
  ax.set_xlabel(compLabels[i], fontsize=ergoPlot.fs_latex)
  ax.set_ylabel(compLabels[j], fontsize=ergoPlot.fs_latex)
  plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
  plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)


