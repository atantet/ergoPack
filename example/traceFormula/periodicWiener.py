import numpy as np

dt = 0.01
time = np.arange(0., 15., dt)
nt = time.shape[0]

n = 100000
x = np.zeros((n,))
m = np.empty((nt,))
v = np.empty((nt,))
m[0] = np.mean(x)
v[0] = np.mean((x - m[0])**2)

for t in np.arange(1, nt):
    for k in np.arange(n):
        x[k] = x[k] + np.sqrt(dt) * numpy.random.randn()
        while x[k] > np.pi:
            x[k] -= 2*np.pi
        while x[k] < -np.pi:
            x[k] += 2*np.pi
            
    m[t] = np.mean(x)
    v[t] = np.mean((x - m[t])**2)

plt.figure()
plt.plot(time, m)
plt.figure()
plt.plot(time, v)
plt.plot(time, np.ones((nt,)) * np.pi**2/3)

