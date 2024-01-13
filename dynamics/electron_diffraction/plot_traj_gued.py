#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_array, issparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider


def load_traj_data_ft(dh, ntraj, prefix):
    ft_shape = dh[f'{prefix}_ft_shape']
    traj_data_ft = []
    for i in range(ntraj):
        data = dh[f'{i:02d}_{prefix}_ft_data']
        row = dh[f'{i:02d}_{prefix}_ft_row']
        col = dh[f'{i:02d}_{prefix}_ft_col']
        data_coo = coo_array((data, (row, col)), shape=ft_shape)
        traj_data_ft.append(data_coo)
    return traj_data_ft


def reconstruct_data(data_ft):
    ft_array = data_ft.toarray()
    data = np.fft.ifft(ft_array, axis=0)
    data = np.fft.irfft(data, axis=1)
    return data


def calculate_avg_traj_data(traj_data_ft):
    avg_data_ft = sum(traj_data_ft) / len(traj_data_ft)
    avg_data = reconstruct_data(avg_data_ft)
    return avg_data


def plot_gued(fig, axs, delta_sm_data, delta_pdf_data, lims, cmap):
    assert len(axs) == 2

    delta_sm, delta_sm_max, sm_extent = delta_sm_data
    delta_pdf, delta_pdf_max, pdf_extent = delta_pdf_data
    tlim, slim, rlim = lims

    if issparse(delta_sm):
        delta_sm = reconstruct_data(delta_sm)
    if issparse(delta_pdf):
        delta_pdf = reconstruct_data(delta_pdf)

    im_sm = axs[0].imshow(
        delta_sm.T, origin='lower', aspect='auto', extent=sm_extent, 
        vmin=-delta_sm_max, vmax=delta_sm_max, cmap=cmap,
    )
    axs[0].set_xlim(tlim)
    axs[0].set_ylim(slim)
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('s / $\AA^{-1}$')
    fig.colorbar(im_sm, ax=axs[0], label='$\Delta sM(s)$ / a.u.')

    im_pdf = axs[1].imshow(
        delta_pdf.T, origin='lower', aspect='auto', extent=pdf_extent,
        vmin=-delta_pdf_max, vmax=delta_pdf_max, cmap=cmap,
    )
    axs[1].set_xlim(tlim)
    axs[1].set_ylim(rlim)
    axs[1].set_xlabel('time / fs')
    axs[1].set_ylabel('r / $\AA$')
    fig.colorbar(im_pdf, ax=axs[1], label='$\Delta P(r)$ / a.u.')
    
    return im_sm, im_pdf

def main():
    GUED_NPZ = 'gued_compressed.npz'
    CMAP_MAX_SCALE = 0.9
    TRAJ_SM_CMAP_MAX_FACTOR = 1.5
    TRAJ_PDF_CMAP_MAX_FACTOR = 2.0

    with np.load(GUED_NPZ) as data:
        ntraj = data['ntraj']
        s_array = data['s_array']
        r_array = data['r_array']
        time_array = data['time_array']
        ds = data['ds']
        dr = data['dr']
        dt = data['dt']
        sm_extent = data['sm_extent']
        pdf_extent = data['pdf_extent']
        cmap = ListedColormap(data['cmap'])
        traj_delta_sm_ft = load_traj_data_ft(data, ntraj, 'delta_sm')
        traj_delta_pdf_ft = load_traj_data_ft(data, ntraj, 'delta_pdf')
    
    lims = (
        (time_array.min(), time_array.max()),
        (s_array.min(), s_array.max()),
        (r_array.min(), r_array.max()),
    )

    avg_delta_sm = calculate_avg_traj_data(traj_delta_sm_ft)
    avg_delta_pdf = calculate_avg_traj_data(traj_delta_pdf_ft)
    
    avg_delta_sm_max = np.max(np.abs(avg_delta_sm)) * CMAP_MAX_SCALE
    avg_delta_pdf_max = np.max(np.abs(avg_delta_pdf)) * CMAP_MAX_SCALE
    delta_sm_max = avg_delta_sm_max * TRAJ_SM_CMAP_MAX_FACTOR
    delta_pdf_max = avg_delta_pdf_max * TRAJ_PDF_CMAP_MAX_FACTOR

    fig1, axs1 = plt.subplots(2, 1, figsize=(8, 6))
    delta_sm_data = (avg_delta_sm, avg_delta_sm_max, sm_extent)
    delta_pdf_data = (avg_delta_pdf, avg_delta_pdf_max, pdf_extent)

    im_sm, im_pdf = plot_gued(fig1, axs1, delta_sm_data, delta_pdf_data, lims, cmap)
    fig1.tight_layout()


    fig2, axs2 = plt.subplots(2, 1, figsize=(8, 6.5))
    delta_sm_data = (traj_delta_sm_ft[0], delta_sm_max, sm_extent)
    delta_pdf_data = (traj_delta_pdf_ft[0], delta_pdf_max, pdf_extent)
    im_sm, im_pdf = plot_gued(fig2, axs2, delta_sm_data, delta_pdf_data, lims, cmap)
    fig2.tight_layout(rect=[0.00, 0.06, 1.00, 1.00])

    # Slider
    axtraj = fig2.add_axes([0.15, 0.03, 0.65, 0.03])
    straj = Slider(
        axtraj, 'Traj ', 0, ntraj - 1, valinit=0, valfmt='traj-%02d',
    )

    # Update function
    def update(val):
        idx = int(straj.val)
        im_sm.set_data(reconstruct_data(traj_delta_sm_ft[idx]).T)
        im_pdf.set_data(reconstruct_data(traj_delta_pdf_ft[idx]).T)
        fig2.canvas.draw_idle()
    
    # Connect the update function to the slider
    straj.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()

