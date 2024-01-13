#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

with np.load('spin_adiabatic_populations.npz') as data:
    ntraj = data['ntraj']
    nstep = data['nstep']
    nstate_sad = data['nstate']
    time = data['time']

    sad_pop = np.zeros((ntraj, nstep, nstate_sad))
    for i in range(ntraj):
        # data[f'traj_{i:02d}'] contains the spin-adiabatic populations of 
        # trajectory i, format (nstep, nstate_sad)
        # state order: |0>, |1>, |2>, |3>, |4>, |5>, |6>, |7>, |8>
        sad_pop[i] = data[f'traj_{i:02d}']

with np.load('spin_diabatic_populations.npz') as data:
    ntraj = data['ntraj']
    nstep = data['nstep']
    nstate_sd = data['nstate']
    time = data['time']

    sd_pop = np.zeros((ntraj, nstep, nstate_sd))
    for i in range(ntraj):
        # data[f'traj_{i:02d}'] contains the spin-diabatic populations of 
        # trajectory i, format (nstep, nstate_sd)
        # The spin-diabatic population at t = 0 is estimated using the 
        # transformation matrix at the optimized geometry.
        # state order: S0, S1, S2, T1, T2
        sd_pop[i] = data[f'traj_{i:02d}']

avg_sad_pop = np.mean(sad_pop, axis=0)
avg_sd_pop = np.mean(sd_pop, axis=0)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
axs[0].plot(time, avg_sad_pop)
axs[1].plot(time, avg_sd_pop)
fig.tight_layout()
plt.show()
