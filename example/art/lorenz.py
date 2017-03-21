import numpy as np

#omega = 1.
#epsilon = 0.3

mu = 1.
gamma = 1.
beta = 0.4
eps = 0.5
omega = gamma - beta*np.sqrt(mu)

nev = 10
nevPlot = 3.5

nw = 1000
nt = 1000

eigVal = np.zeros((nev*2+1,), dtype=complex)
xticks = np.zeros((nev*2+1,))
yticks = np.zeros((int(nevPlot),))
xticklabels = [r'$0$']*(nev*2+1)
yticklabels = ['']*int(nevPlot)
for k in np.arange(1,nev*2+1):
    n = int(np.ceil(k*1./2))
    eigVal[k] = -n**2 * eps**2 * (1 + beta**2) / (2*mu)
    if k%2 == 0:
        eigVal[k] += 1j * n * omega
        xticklabels[nev+n] = r'$\Im(\lambda_{%d})$' % n
        xticks[nev+n] = n*omega
    else:
        eigVal[k] -= 1j * n * omega
        xticklabels[nev-n] = r'$-\Im(\lambda_{%d})$' % n
        xticks[nev-n] = -n*omega
        

#weights = np.ones((eigVal.shape[0],))
weights = (2)**(-np.ceil(np.arange(eigVal.shape[0])*1./2))
weights /= weights.sum()
weights[1:3] /= 2

# Get correlation
time = np.linspace(0, -4. / eigVal[1].real, nt)
C = np.zeros((nt,), dtype=complex)
for k in np.arange(1, eigVal.shape[0]):
    C += weights[k] * np.exp(eigVal[k]*time)
C = C.real
C /= C[0]

# Get power spectrum
angFreq = np.linspace(-nevPlot*omega, nevPlot*omega, nw)
S = np.zeros((nw,))
for k in np.arange(eigVal.shape[0]):
    S += (-eigVal[k].real / ((angFreq - eigVal[k].imag)**2 + eigVal[k].real**2) * weights[k] / np.pi).real
# for k in np.arange(1, int(nevPlot)+1):
#     yticks[1-k] = S[np.argmin(np.abs(angFreq - eigVal[(k-1)*2+1].imag))].real / 2
#     yticklabels[1-k] = r'$S(\Im\lambda_%d) / 2$' % k
yticks = []
yticklabels = []

fig = plt.figure(figsize=[12, 6])
ax = fig.add_subplot(111)
ax.plot(angFreq, S, '-k', linewidth=2)
ax.set_yscale('log', basey=10)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize='xx-large')
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize='xx-large')
ax.set_xlim(angFreq[0], angFreq[-1])
ax.set_ylim(S[nw/2+1]*0.9, S[np.argmin(np.abs(angFreq - eigVal[1].imag))].real*1.5)
ax.spines['left'].set_position('center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.minorticks_off()
ax.set_xlabel(r'$\omega$', fontsize='xx-large')
ax.set_ylabel(r'$S(\omega)$', fontsize='xx-large')
# Half width
keig = 4
xa = eigVal[keig].imag + eigVal[keig]
ya = S[np.argmin(np.abs(angFreq - eigVal[keig].imag))] / 2
dxa = -eigVal[keig].real * 2
dya = 0
#ax.arrow(xa, ya, dxa, dya, head_width=0.05, head_length=0.1, fc='k', ec='k')
ax.annotate(s='', xy=(xa+dxa,ya+dya), xytext=(xa,ya),
            arrowprops=dict(arrowstyle='<->', mutation_scale=20))
ax.text((xa+dxa)*1.01, ya, r'$\Delta \omega_{1/2} = 2 |\Re(\lambda_2)|$', fontsize='x-large')
keig = 2
xa = (eigVal[keig].imag)*1
ya = S[np.argmin(np.abs(angFreq - eigVal[keig].imag))]
ax.text(xa, ya, r'$S(\Im(\lambda_1)) = \frac{w_1}{\pi \Re(\lambda_1)}$', fontsize='x-large')
ax.plot([0., xa], [ya,ya], '--k')

# Plot correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time, C.real, '-k', linewidth=2)
ax.set_xlim(time[0], time[-1])
ax.set_ylim(-1, 1)
ax.set_xlabel(r'$t$', fontsize='xx-large')
ax.set_ylabel(r'$C_{xx}(t)$', fontsize='xx-large')
fig.savefig('plot/lorentz_ccf.eps', dpi=300, bbox_inches='tight')
