import numpy as np
import matplotlib.pyplot as plt

eigLin = np.array([-0.2 + 1j * 2*np.pi/8, -0.2 - 1j * 2*np.pi/8],
                  dtype=complex)
dim = 2

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
msize = 128
#            figFormat = 'eps'
figFormat = 'png'
dpi = 300
readMap = False
xminEigval = -1.
yminEigval = -4.5
nevLin = 10

fig = plt.figure()
ax = fig.add_subplot(111)
eigs = np.zeros((nevLin**dim,), dtype=complex)
for k in np.arange(nevLin**dim):
    subbp = k
    strg = ''
    for d in np.arange(dim):
        subbn = int(subbp/nevLin)
        ids = subbp - subbn*nevLin # Index of the box in each direction ind
        eigs[k] += ids * eigLin[d]
        subbp = subbn
        strg += 'k%d = %d' % (d, ids)
    ax.scatter(eigs[k].real, eigs[k].imag, c='k', s=msize, marker='o')
    ax.scatter(0., 1./0.4 * eigLin[0].imag*k, c='b', s=msize, marker='o')
    ax.scatter(0., -1./0.4 * eigLin[0].imag*k, c='b', s=msize, marker='o')
ax.scatter(0., 0, c='r', s=msize, marker='o')

ax.set_xlabel(r'$\Re(\lambda_i)$', fontsize=fs_latex)
ax.set_ylabel(r'$\Im(\lambda_i)$', fontsize=fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
#ax.set_title('%d-time-step spectrum for %s\nSlowest time-scale: %.1f' \
    #    % (tau, srcPostfix, -1. / rate[0]))
ax.set_xlim(xminEigval, -xminEigval / 10)
ax.set_ylim(yminEigval, -yminEigval)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
p = plt.axvspan(xminEigval, 0.9 * xminEigval, facecolor='k')
ax.plot([0., 0.], ylim, '--k')
fig.savefig('artSpectrum.%s' % figFormat, bbox_inches='tight', dpi=dpi)

