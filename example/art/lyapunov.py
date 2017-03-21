import numpy as np

muSup = 1.
eps = 1.

Rpo = np.sqrt(muSup)

rmax = 1.7 * Rpo
r = np.linspace(0.001, rmax, 1001)

def U(r, mu, eps=0.):
    if mu > 0:
        Ur = + mu**2/4 - mu*r**2/2 + r**4/4 - eps**2/2*np.log(r)
    else:
        Ur = - mu*r**2/2 + r**4/4 - eps**2/2*np.log(r)
    return Ur
def dU(r, mu, eps=0.):
    return -mu*r + r**3 - eps**2/(2*r)
def ddU(r, mu, eps=0.):
    return -mu + 3*r**2 + eps**2/(2*r**2)

def KU(r, mu, eps=0.):
    return -dU(r, mu, eps)**2 + eps**2/2*ddU(r, mu, eps)

def KU0(r, mu, eps=0.):
    return -dU(r, mu)**2 \
        + eps**2/(2*r) * dU(r, mu) \
        + eps**2/2 * ddU(r, mu)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(r, U(r, muSup, eps))
# ax.plot(r, KU(r, muSup, eps))
# ax.set_xlim(0., rmax)
# ax.set_ylim(-2, 2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(r, U(r, muSup))
# ax.plot(r, KU0(r, muSup, eps))
# ax.set_xlim(0., rmax)
# ax.set_ylim(-2, 2)

# a = 1.6
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(r, a*U(r, muSup))
# ax.plot(r, KU0(r, muSup, eps))
# ax.set_xlim(0., rmax)
# ax.set_ylim(-2, 2)

# alpha = 1.
# beta = 2.
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(r, -alpha * U(r, muSup) + beta)
# ax.plot(r, KU0(r, muSup, eps))
# ax.set_xlim(0., rmax)
# ax.set_ylim(-2, 2)




fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r, (U(r, 0.) + 1))
ax.plot(r, KU0(r, 0., eps))
ax.set_xlim(0., rmax)
ax.set_ylim(-2, 2)

a = 1.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r, a*(U(r, 0.) + 1))
ax.plot(r, KU0(r, 0., eps))
ax.set_xlim(0., rmax)
ax.set_ylim(-2, 2)

alpha = 1.
beta = 3
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r, -alpha * (U(r, 0.) + 1) + beta)
ax.plot(r, KU0(r, 0., eps))
ax.set_xlim(0., rmax)
ax.set_ylim(-2, 2)
