import os
import numpy as np
import matplotlib.pyplot as plt
import atmath

os.system('mkdir sim 2> /dev/null')
levels = 20
fs_default = 'x-large'
fs_latex = 'xx-large'
fs_xlabel = fs_default
fs_ylabel = fs_default
fs_xticklabels = fs_default
fs_yticklabels = fs_default
fs_legend_title = fs_default
fs_legend_labels = fs_default
fs_cbar_label = fs_default
#            figFormat = 'eps'
figFormat = 'png'
dpi = 300

A = np.array([[-0.4]])
Q = np.array([[.025]])

sim0 = np.array([0., 0.5, 1.5])
mu = 1

# Duration
spinup = 0
plotL = 100
spinupCCF = 50
L = 1e5
lagMax = 50
dt = 0.01
sampling = 1

dim = A.shape[0]
time = np.arange(0, L, dt * sampling)
nt = time.shape[0]
nStep = L / dt
nStepSpinup = spinup / dt
nStepSpinupCCF = spinupCCF / (dt * sampling)
nStepPlotL = plotL / (dt * sampling)
nStepLagMax = lagMax / (dt * sampling)

# Integration
print 'Integrating...'
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
simFast = np.zeros((sim0.shape[0], nt - nStepSpinup / sampling))
for s in np.arange(sim0.shape[0]):
    state = np.ones((dim,)) * sim0[s]
    for i in np.arange(0, nStep):
        newState = state + dt * np.dot(A, state - mu) \
                   + np.sqrt(dt) * np.dot(Q, np.random.normal(loc=0.0,
                                                              scale=1.0,
                                                              size=dim))
        state = newState
        if (i >= nStepSpinup) & ((i - nStepSpinup)%sampling == 0):
            simFast[s, (i - nStepSpinup) / sampling] = state

    ax.plot(time[:nStepPlotL], simFast[s][:nStepPlotL], linewidth=2)
ax.set_ylabel(r'$g(S_t x)$', fontsize=fs_latex)
ax.set_xlim(time[0], plotL)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)

# Integration
sim0 = np.array([0., 0.5, 1.5]) * 2
print 'Integrating...'
simSlow = np.zeros((sim0.shape[0], nt - spinup / (dt * sampling)))
for s in np.arange(sim0.shape[0]):
    state = np.ones((dim,)) * sim0[s]
    for i in np.arange(0, nStep):
        newState = state + dt * np.dot(A / 3, state - mu * 2) \
                   + np.sqrt(dt) * np.dot(Q, np.random.normal(loc=0.0,
                                                              scale=1.0,
                                                              size=dim))
        state = newState
        if (i >= nStepSpinup) & ((i - nStepSpinup)%sampling == 0):
            simSlow[s, (i - nStepSpinup) / sampling] = state

    ax2.plot(time[:nStepPlotL], simSlow[s][:nStepPlotL], linewidth=1, linestyle='--')

ax2.set_ylabel(r'$h(S_t x)$', fontsize=fs_latex)
ax.set_ylim(0., 3.)
ax2.set_ylim(0., 3.)
plt.setp(ax2.get_yticklabels(), fontsize=fs_yticklabels)
ax.set_xlabel(r'$t$', fontsize=fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
fig.savefig('spinupOU.%s' % figFormat, bbox_inches='tight', dpi=dpi)

# Plot averages
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
for s in np.arange(sim0.shape[0]):
    ax.plot(time[:nStepPlotL], (np.cumsum(simFast[s]) / np.arange(1, simFast[0].shape[0]+1))[:nStepPlotL], linewidth=2)
ax.set_ylabel(r'$\frac{1}{T} \int_0^T g(S_t x) dt$', fontsize=fs_latex)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
for s in np.arange(sim0.shape[0]):
    ax2.plot(time[:nStepPlotL], (np.cumsum(simSlow[s]) / np.arange(1, simSlow[0].shape[0]+1))[:nStepPlotL], linewidth=1, linestyle='--')
ax2.set_ylabel(r'$\frac{1}{T} \int_0^T h(S_t x) dt$', fontsize=fs_latex)
ax.set_ylim(0., 3.)
ax2.set_ylim(0., 3.)
plt.setp(ax2.get_yticklabels(), fontsize=fs_yticklabels)
ax.set_xlim(time[0], plotL)
ax.set_xlabel(r'$T$', fontsize=fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
fig.savefig('spinupOUAvg.%s' % figFormat, bbox_inches='tight', dpi=dpi)

# Print getting ccf
print 'Getting CCF...'
ccfFast = atmath.ccf(simFast[1][nStepSpinupCCF:], simFast[1][nStepSpinupCCF:],
                     lagMax=nStepLagMax)[nStepLagMax:-1]
ccfSlow = atmath.ccf(simSlow[1][nStepSpinupCCF:], simSlow[1][nStepSpinupCCF:],
                     lagMax=nStepLagMax)[nStepLagMax:-1]

# Plotting ccf
fig = plt.figure()
ax = fig.add_subplot(111)
lags = np.arange(0, lagMax, dt * sampling)
pt1 = ax.plot(lags, np.exp(A[0, 0] * lags), 'k-', linewidth=2,
              label=r'$C_{g, g}(t)$')
pt2 = ax.plot(lags, np.exp(A[0, 0] / 3 * lags), 'k--', linewidth=1,
              label=r'$C_{h, h}(t)$')
ax.set_xlim(0, lagMax)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r'$t$', fontsize=fs_latex)
#ax.set_ylabel(r'$C_{g, g}, C_{h, h}(t)(t)$', fontsize=fs_latex)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels) 
ax.legend(loc='upper right', frameon=False, fontsize=fs_latex)
fig.savefig('spinupOUCCF.%s' % figFormat, bbox_inches='tight', dpi=dpi)
