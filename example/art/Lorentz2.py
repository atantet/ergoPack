import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', **{'family': 'serif'})

omega = 1.
epsilon = 0.3

nev = 10
nevPlot = 3.5

nw = 1000

eigVal = np.zeros((nev*2+1,), dtype=complex)
xticklabelsEig = ['']*int(nevPlot)
yticklabelsEig = [r'$0$']*(nev*2+1)
xticksEig = np.zeros((int(nevPlot),))
yticksEig = np.zeros((nev*2+1,))
for k in np.arange(1, nev*2+1):
    n = int(np.ceil(k*1./2))
    eigVal[k] = -n**2 * epsilon**2 / 2
    if k % 2 == 0:
        eigVal[k] += 1j * n * omega
        yticklabelsEig[nev+n] = r'$\Im(\lambda_{%d})$' % n
        yticksEig[nev+n] = eigVal[k].imag
    else:
        eigVal[k] -= 1j * n * omega
        yticklabelsEig[nev-n] = r'$-\Im(\lambda_{%d})$' % n
        yticksEig[nev-n] = eigVal[k].imag


# weights = np.ones((eigVal.shape[0],))
weights = (2)**(-np.ceil(np.arange(eigVal.shape[0])*1./2))
weights /= weights.sum()
weights[1:3] /= 2

angFreq = np.linspace(-nevPlot*omega, nevPlot*omega, nw)
S = np.zeros((nw,))
for k in np.arange(eigVal.shape[0]):
    S += (-eigVal[k].real / ((angFreq - eigVal[k].imag) **
                             2 + eigVal[k].real**2) * weights[k] / np.pi).real

for k in np.arange(1, int(nevPlot)+1):
    xticksEig[1-k] = eigVal[(k-1)*2+1].real
    xticklabelsEig[1-k] = r'$\Re(\lambda_%d)$' % k
    # xticksEig[1-k] = S[np.argmin(np.abs(
    #     angFreq - eigVal[(k-1)*2+1].imag))].real / 2
    # xticklabelsEig[1-k] = r'$S(\Im\lambda_%d) / 2$' % k


msize = 32
xlimEig = [(eigVal[6].real+eigVal[7].real)/2, 0.01]
ylimEig = [(eigVal[5].imag+eigVal[7].imag)/2,
           -(eigVal[5].imag+eigVal[7].imag)/2]


# Create axis for eigenvalue and power spectrum panels
nullfmt = plt.NullFormatter()
leftEig, widthEig = 0.1, 0.25
bottomEig, heightEig = 0.1, 0.9
leftPow = leftEig + widthEig + 0.08
bottomPow = 0.1
heightPow = 0.9
widthPow = 0.25

rectEig = [leftEig, bottomEig, widthEig, heightEig]
rectPow = [leftPow, bottomEig, widthPow, heightEig]
ratio = heightEig / (widthEig*2) * 1.2
defaultFigHeight = plt.rcParams['figure.figsize'][1]
fig = plt.figure(figsize=(defaultFigHeight * ratio, defaultFigHeight))
axEig = plt.axes(rectEig)
axPowLeft = plt.axes(rectPow)
axPow = axPowLeft.twinx()

axEig.grid()
axEig.scatter(eigVal.real, eigVal.imag, c='k',
              s=msize, marker='o', edgecolors='face')
axEig.set_xlabel(r'$\Re(\lambda_k)$', fontsize='xx-large')
axEig.set_ylabel(r'$\Im(\lambda_k)$', fontsize='xx-large')
axEig.set_xticks(xticksEig)
axEig.set_xticklabels(xticklabelsEig)
axEig.set_yticks(yticksEig)
axEig.set_yticklabels(yticklabelsEig)
axEig.set_xlim(xlimEig)
axEig.set_ylim(ylimEig)
axEig.invert_yaxis()
# axEig.spines['bottom'].set_position('center')
axEig.spines['top'].set_visible(False)
axEig.spines['left'].set_visible(False)
axEig.yaxis.tick_right()
plt.setp(axEig.get_xticklabels(), fontsize='x-large')
plt.setp(axEig.get_yticklabels(), fontsize='x-large')

# Add complex plane symbol
xlim = axEig.get_xlim()
dx = xlim[1] - xlim[0]
ylim = axEig.get_ylim()
dy = ylim[1] - ylim[0]
axEig.text(xlim[0] + 0.05 * dx, ylim[1] - 0.18 * dy, r'$\mathbb{C}$',
           fontsize=28)

axPow.plot(S, angFreq, '-k', linewidth=2)
axPow.set_xscale('log', basex=10)
axPowLeft.set_xscale('log', basex=10)
axPow.set_yticks(yticksEig)
axPowLeft.set_yticks(yticksEig)
axPow.set_yticklabels([])
axPowLeft.set_yticklabels([])
axPow.set_ylim(angFreq[0], angFreq[-1])
axPowLeft.set_ylim(angFreq[0], angFreq[-1])
axPow.set_xlim(
    S[nw // 2 + 1]*0.9, S[np.argmin(np.abs(angFreq - eigVal[1].imag))].real*2.)
# axPow.spines['bottom'].set_position('center')
axPow.spines['left'].set_visible(False)
axPow.spines['top'].set_visible(False)
# axPowLeft.spines['left'].set_visible(False)
axPow.get_xaxis().tick_bottom()
axPow.get_yaxis().tick_left()
plt.minorticks_off()
axPowLeft.set_xlabel(r'$S(\omega)$', fontsize='xx-large')
axPow.set_ylabel(r'$\omega$', fontsize='xx-large')
# Half width
keig = 4
ya = eigVal[keig].imag + eigVal[keig] - 0.05
xa = S[np.argmin(np.abs(angFreq - eigVal[keig].imag))] / 2
dya = -eigVal[keig].real * 2 + 2 * 0.05
dxa = 0
# axPow.arrow(xa, ya, dxa, dya, head_width=0.05, head_length=0.1,
#             fc='k', ec='k')
axPow.annotate(s='', xy=(xa+dxa, ya+dya), xytext=(xa, ya),
               arrowprops=dict(arrowstyle='<->', mutation_scale=16))
axPow.text(xa*0.5, (ya+dya)*1.18,
           r'$\Delta \omega_{1/2} \approx 2 |\Re(\lambda_2)|$',
           fontsize='x-large')
keig = 2
ya = (eigVal[keig].imag)*1
xa = S[np.argmin(np.abs(angFreq - eigVal[keig].imag))]
text_max = r'$S(\Im(\lambda_1)) \approx a_1 / (\pi \Re(\lambda_1))$'
xlim = axPow.get_xlim()
axPow.set_xticks([xlim[0], xa])
axPow.set_xticklabels(['0', text_max])
axPow.invert_yaxis()
plt.setp(axPowLeft.get_xticklabels(), fontsize='x-large')
axPowLeft.grid()

fig.savefig('lorentz.eps', dpi=300, bbox_inches='tight')

plt.show(block=False)
