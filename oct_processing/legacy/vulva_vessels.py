import numpy as np
#import re
import os
#import xlwt
from scipy import signal, stats, spatial, special
from skimage import morphology, measure, filters
from skimage.restoration import inpaint
import pandas as pd
import matplotlib.pyplot as plt

from importlib import reload


from oct_processing import brain_2019 as br
#from oct_processing import depth_resolved as dr
from oct_processing import optical_properties as op
from oct_processing import oct_processing as oct_p

reload(op)
reload(oct_p)


def get_3d_distributions(filename):

    #array = br.import_SU_bin(filename, (1024, 256, 256), np.complex64)
    co = br.get_array_tuple(filename)

    '''
    array_vess = np.zeros(array[::2, :, :].shape)+1j*np.zeros(array[::2, :, :].shape)

    for i in range(16):
        array_vess[:, :, i::16] = br.real_cumul_angle(
            array[::2, :, i::16], N=80, amplitude=False)

    vess = np.zeros(array_vess[:, :, :].shape)+1j*np.zeros(array_vess[:, :, :].shape)

    print('vasc_process')
    for i in range(16):
        vess[:, :, i::16] = br.high_pass_nonlin(array_vess[:, :, i::16], 7, 2*424, 3, False, .5)
    '''

    array, vess = oct_p.get_vessels_from_filepath(filename)

    array_ = signal.fftconvolve(np.abs(array[1::2, :, :])**2, np.ones((5, 5, 5)), 'same')/5/5/5
    #array_ = signal.fftconvolve(np.abs(array)**2, np.ones((5, 5, 5)), 'same')/5/5/5
    att_cross = op.get_depth_resolved_attenuation(np.abs(array_), Nwind=32)
    att_cross = att_cross/.01

    array = signal.fftconvolve(np.abs(array[::2, :, :])**2, np.ones((5, 5, 5)), 'same')/5/5/5
    #array = signal.fftconvolve(np.abs(array)**2, np.ones((5, 5, 5)), 'same')/5/5/5
    att_co, snr = op.get_depth_resolved_attenuation(np.abs(array), Nwind=32, return_snr=True)
    att_co = att_co/.01

    dmtrt = 24  # 16
    bg = signal.fftconvolve(att_co, np.ones((dmtrt, dmtrt, dmtrt)), 'same')/dmtrt/dmtrt/dmtrt
    aux = (att_co-bg)**2
    aux[aux > 48] = 48  # 32
    var = signal.fftconvolve(aux, np.ones((dmtrt, dmtrt, dmtrt)), 'same')/dmtrt/dmtrt/dmtrt

    rolled, supp = op.image_roll_3d(
        (co, att_co, att_cross, np.abs(vess), var, snr), 210)  # 200)

    lymph = (rolled[-1] > 10)*(rolled[1] < 2.)*(rolled[1] > .0)*(rolled[-2] > 8.)

    lymph = morphology.remove_small_objects(lymph > 0, 200)
    lbls = measure.label(lymph > 0)

    artifact_ind = np.unique(lbls[:, :11, :])

    clean_ind = np.isin(np.unique(lbls)[1:], artifact_ind[1:], assume_unique=False, invert=True)
    clean_ind = np.unique(lbls)[1:][clean_ind]

    lymph_clean = np.zeros(lymph.shape)

    for i in clean_ind:

        lymph_clean += lbls == i

    return (rolled[1], rolled[2], rolled[3], lymph_clean, att_co, att_cross)


def get_maps(distr_tuple,
             depth_tuple,
             threshold=.1,
             min_vess_size=100):

    att_co = distr_tuple[0]
    att_cross = distr_tuple[1]
    lymph = distr_tuple[2]
    vess = distr_tuple[3]
    att_co_unshifted = distr_tuple[4]
    att_cross_unshifted = distr_tuple[5]

    att_co_arr = np.zeros((np.int(att_co.shape[0]/2), att_co.shape[-1], len(depth_tuple)-1))
    att_cross_arr = np.zeros(
        (np.int(att_cross.shape[0]/2), att_cross.shape[-1], len(depth_tuple)-1))
    lymph_arr = np.zeros((np.int(lymph.shape[0]/2), lymph.shape[-1], len(depth_tuple)-1))
    vess_arr = np.zeros((np.int(vess.shape[0]/2), vess.shape[-1], len(depth_tuple)-1))
    vess_mip_arr = np.zeros((np.int(vess.shape[0]/2), vess.shape[-1], len(depth_tuple)-1))

    for i in range(len(depth_tuple)-1):

        d = (depth_tuple[i+1]+depth_tuple[i])/2.
        N = np.int(d/6.)
        h = 8
        Nmin = max(0, N-h)
        Nmax = min(N+h, att_co.shape[0])
        # rolled[4][::2, Nmin:Nmax, :]
        # aux = filters.gaussian(np.mean(lymph[:, Nmin:Nmax, :], axis=1), 1)

        # seed = morphology.erosion(aux, morphology.disk(8))  # aux/2.  # np.copy(aux)
        # seed[1:-1, 1:-1] = aux.min()
        # mask = aux
        # mask[mask < seed] = seed[mask < seed]

        # dil_bg = morphology.reconstruction(seed, mask, method='dilation',
        #                                   selem=morphology.disk(1))  # np.ones((1, 1)))

        lymph_map = np.mean(lymph[::2, np.int(depth_tuple[i]/6)
                            :np.int(depth_tuple[i+1]/6), :], axis=1)  # aux-dil_bg
        lymph_map = signal.resample(lymph_map[:, 30:-30], lymph_map.shape[1], axis=1)

        #D = 1
        #vess = signal.fftconvolve(vess, np.ones((D, D, D)), 'same')/D/D/D

        #aux = calculate_l3(vess[::2, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :])
        #aux,_ = stats.normaltest(vess[::2, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :],axis=1)
        vess_mip_arr[:, :, i] = np.max(
            vess[::2, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :], axis=1)
        aux = vess_mip_arr[:, :, i] - \
            np.median(vess[::2, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :], axis=1)

        aux_ = filters.gaussian(aux, 1)

        print('max_aux', np.max(aux), np.median(aux))

        # seed = morphology.erosion(aux, morphology.disk(8))  # aux/2.  # np.copy(aux)
        # seed[1:-1, 1:-1] = aux.min()
        # mask = aux
        # mask[mask < seed] = seed[mask < seed]

        # dil_bg = morphology.reconstruction(seed, mask, method='dilation',
        #                                   selem=morphology.disk(1))  # np.ones((1, 1)))

        D = 128
        # /D/D  # np.mean(aux-dil_bg, axis=1)
        bg = signal.fftconvolve(aux_, np.ones((1, D)), 'same')
        norm = signal.fftconvolve(np.ones(aux.shape), np.ones((1, D)), 'same')
        bg = bg/norm

        aux = aux_-1*bg  # [:, np.newaxis]
        aux[aux < .0] = 10**-19
        aux /= np.percentile(aux, 95)  # .max()
        bg /= np.percentile(bg, 95)  # .max()
        vess_map = aux/(1.+(1.*bg/aux)**2)
        vess_map = signal.resample(vess_map[:, 20:-20], vess_map.shape[1], axis=1)
        vess_map = signal.resample(vess_map[20:-20, :], vess_map.shape[0], axis=0)
        #vess_map = np.zeros(vess_map.shape)

        vess_mask = morphology.remove_small_objects(vess_map > threshold, min_vess_size)
        vess_map = vess_map*vess_mask

        mask = np.sum(vess_map, axis=1) > 20
        mask = mask[:, np.newaxis]
        mask = np.ones(vess_map.shape)*mask
        mask = morphology.binary_dilation(mask > 0, np.ones((1, 1)))
        #mask[:10,:] = 0
        #mask[-10:,:] = 0
        #vess_map = vess_map*(mask==0)

        #vess_map = inpaint.inpaint_biharmonic(1.*vess_map, 1.*mask)

        att_co_map = np.mean(att_co[::2, Nmin:Nmax, :], axis=1)
        att_cross_map = np.mean(att_cross[::2, Nmin:Nmax, :], axis=1)
        att_diff_map = np.mean(att_co[::2, Nmin:Nmax, :]-att_cross[::2, Nmin:Nmax, :], axis=1)

        att_co_map = signal.resample(att_co_map[:, 30:-30], att_co_map.shape[1], axis=1)
        att_cross_map = signal.resample(att_cross_map[:, 30:-30], att_cross_map.shape[1], axis=1)
        att_diff_map = signal.resample(att_diff_map[:, 30:-30], att_diff_map.shape[1], axis=1)

        att_co_arr[:, :, i] = att_co_map
        att_cross_arr[:, :, i] = att_cross_map
        lymph_arr[:, :, i] = lymph_map
        vess_arr[:, :, i] = vess_map

    return (att_co_arr, att_cross_arr, lymph_arr, vess_arr, vess_mip_arr,
            (att_co_unshifted[200, :, :], att_co_unshifted[500, :, :],
             att_co_unshifted[-200, :, :]),
            (att_cross_unshifted[200, :, :], att_cross_unshifted[500, :, :], att_cross_unshifted[-200, :, :]))


def calculate_l3(volume):

    sorted = np.sort(volume, axis=1)

    l3 = np.zeros((volume.shape[0], volume.shape[-1]))
    l2 = np.zeros((volume.shape[0], volume.shape[-1]))

    N = volume.shape[1]

    for i in range(N):

        coeff_l3 = special.binom(i, 2)-2*special.binom(i, 1) * \
            special.binom(N-i-1, 1)+special.binom(N-i-1, 1)
        l3 += coeff_l3*sorted[:, i, :]
        coeff_l2 = special.binom(i, 1)-special.binom(N-i-1, 1)
        l2 += coeff_l2*sorted[:, i, :]

    l3 = l3/3/special.binom(N, 3)
    l2 = l2/2/special.binom(N, 2)

    return (l2/(np.abs(l3)+.2))  # /(l2+.0001))


def get_vessels_thickness(bnr_distr):

    if np.sum(bnr_distr) > 0:
        if len(bnr_distr.shape) == 2:
            kernel = morphology.disk(1)
        else:
            kernel = morphology.ball(1)

        # ^(morphology.binary_erosion(bnr_distr,kernel))
        border = (morphology.binary_dilation(bnr_distr, kernel)) ^ bnr_distr
        # skeleton = morphology.skeletonize(bnr_distr)

        if len(bnr_distr.shape) == 2:

            skeleton = morphology.skeletonize(bnr_distr)

            x = np.arange(bnr_distr.shape[1])
            y = np.arange(bnr_distr.shape[0])
            X, Y = np.meshgrid(x, y)

            vec_skel = np.append(X[skeleton][:, np.newaxis], Y[skeleton][:, np.newaxis], axis=1)
            vec_border = np.append(X[border][:, np.newaxis], Y[border][:, np.newaxis], axis=1)

        else:

            bnr_distr_ = morphology.binary_closing(bnr_distr, morphology.ball(6))
            skeleton = morphology.skeletonize_3d(bnr_distr_)

            x = np.arange(bnr_distr.shape[0])
            z = np.arange(bnr_distr.shape[1])
            y = np.arange(bnr_distr.shape[2])
            X, Y, Z = np.meshgrid(z, x, y)

            vec_skel = np.append(X[skeleton > 0][:, np.newaxis],
                                 Y[skeleton > 0][:, np.newaxis], axis=1)
            vec_skel = np.append(vec_skel, Z[skeleton > 0][:, np.newaxis], axis=1)
            vec_border = np.append(X[border > 0][:, np.newaxis],
                                   Y[border > 0][:, np.newaxis], axis=1)
            vec_border = np.append(vec_border, Z[border > 0][:, np.newaxis], axis=1)

        tree = spatial.cKDTree(vec_skel)
        dist, ind = tree.query(vec_border, 1)

    else:
        dist = np.zeros(10)

    return (dist)


def get_thickness_volume(bnr_distr_3d, N=17):

    bnr_distr_3d_ = morphology.binary_closing(bnr_distr_3d, morphology.ball(6))
    skeleton = morphology.skeletonize_3d(bnr_distr_3d_)
    aux = thinning_aux(bnr_distr_3d_, N)

    return (aux*(skeleton > 0))


def analyze_file(filename,
                 depth_tuple,
                 threshold=.1,
                 min_vess_size=100):

    att_co, att_cross, vess, lymph, att_co_unshifted, att_cross_unshifted = get_3d_distributions(
        filename)
    att_co_arr, att_cross_arr, lymph_arr, vess_arr, vess_mip, co_b_scans, cross_b_scans = get_maps(
        (att_co, att_cross, lymph, vess, att_co_unshifted, att_cross_unshifted), depth_tuple, threshold, min_vess_size)

    lymph = signal.resample(1.*lymph[:, :, 30:-30], lymph.shape[-1], axis=-1)

    lymph_skel_3d = get_thickness_volume(lymph[::2, :, :] > .01, N=17)

    vess_characteristics = np.zeros((len(depth_tuple)-1, 6))
    lymph_characteristics = np.zeros((len(depth_tuple)-1, 6))
    coeff_characteristics = np.zeros((len(depth_tuple)-1, 6))

    for i in range(len(depth_tuple)-1):

        aux_vess = get_vessels_thickness(vess_arr[:, :, i] > .01)
        vess_thick_2d = thinning_aux(vess_arr[:, :, i] > .01, N=7, tp=0)
        skel = morphology.skeletonize(vess_arr[:, :, i] > .01)
        vess_thick_2d = vess_thick_2d*skel
        aux_lymph = lymph_skel_3d[:, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :]
        # aux_lymph_ = aux_lymph[aux_lymph > .0]

        for j in range(1, 5):
            vess_characteristics[i, j-1] = np.sum(aux_vess == j)
            lymph_characteristics[i, j-1] = np.sum(aux_lymph == j)

            # aux = signal.fftconvolve(1.*(vess_thick_2d == j), np.ones((32, 32)), 'same')/32/32
            # vess_characteristics[i, 5+j-1] = np.mean(aux)
            # aux = np.sum(aux_lymph == j, axis=1)
            # aux = signal.fftconvolve(1.*aux, np.ones((32, 32)), 'same')/32/32
            # lymph_characteristics[i, 5+j-1] = np.mean(aux)

        vess_characteristics[i, 4] = np.sum(aux_vess >= 5)
        lymph_characteristics[i, 4] = np.sum(aux_lymph >= 5)

        aux = signal.fftconvolve(1.*(vess_thick_2d > 0), np.ones((32, 32)), 'same')/32/32
        vess_characteristics[i, 5] = np.mean(aux)
        aux = np.sum(aux_lymph > 0, axis=1)
        aux = signal.fftconvolve(1.*aux, np.ones((32, 32)), 'same')/32/32
        lymph_characteristics[i, 5] = np.mean(aux)

        coeff_characteristics[i, 0] = np.mean(att_co_arr[:, :, i])
        coeff_characteristics[i, 1] = np.std(att_co_arr[:, :, i])

        coeff_characteristics[i, 2] = np.mean(att_cross_arr[:, :, i])
        coeff_characteristics[i, 3] = np.std(att_cross_arr[:, :, i])

        coeff_characteristics[i, 4] = np.mean(att_co_arr[:, :, i]-att_cross_arr[:, :, i])
        coeff_characteristics[i, 5] = np.std(att_co_arr[:, :, i]-att_cross_arr[:, :, i])

    vess_characteristics[:, :5] = vess_characteristics[:, :5]/np.prod(vess_arr[:, :, 0].shape)
    lymph_characteristics[:, :5] = lymph_characteristics[:, :5]/np.prod(vess_arr[:, :, 0].shape)

    return (att_co_arr, att_cross_arr, lymph_arr, vess_arr, vess_mip, vess_characteristics, lymph_characteristics, coeff_characteristics,
            co_b_scans, cross_b_scans)


def analyze_folder(foldername,
                   depth_xls,
                   save_xls_path,
                   threshold=.1,
                   min_vess_size=100):

    df = pd.read_excel(depth_xls)

    names = ()

    vess_characteristics = np.zeros((3, 6, 1))
    lymph_characteristics = np.zeros((3, 6, 1))
    coeff_characteristics = np.zeros((3, 6, 1))

    for j in range(df.shape[0]):

        name = df['name'].iloc[j].split('.')[0]  # [:-4]
        print(j, name)
        d0 = df['d0'].iloc[j]
        d1 = df['d1'].iloc[j]
        # d_tuple = (d0/2, d0+(d1-d0)/2, d1+300)  # 75
        d_tuple = (0, d0, d1, d1+200)  # d0/2

        filename = f'{foldername}/{name}.dat'
        print(filename)

        try:

            att_co_arr, att_cross_arr, lymph_arr, vess_arr, vess_mip, vess_characteristics_, lymph_characteristics_, coeff_characteristics_, co_b_scans, cross_b_scans = analyze_file(
                filename, d_tuple, threshold, min_vess_size)

            names = names+(name,)

            directory = f'{foldername}/{name}'
            if os.path.isdir(directory) == False:
                os.mkdir(directory)

            fig, axs = plt.subplots(2, 3, figsize=(10, 10), gridspec_kw={
                                    'wspace': .01, 'hspace': -.3})

            fig.suptitle(name)

            axs[0, 0].imshow(vess_arr[:, :, 0], vmin=0, cmap='hot')  # , vmax=.05,
            axs[0, 0].set_title(f'vessels depth_max={str(d0)}')
            axs[0, 0].xaxis.set_visible(False)
            axs[0, 0].yaxis.set_visible(False)
            axs[0, 1].imshow(vess_arr[:, :, 1], vmin=0, cmap='hot')
            axs[0, 1].set_title(f'vessels depth_max={str(d1)}')
            axs[0, 1].xaxis.set_visible(False)
            axs[0, 1].yaxis.set_visible(False)
            axs[0, 2].imshow(vess_arr[:, :, 2], vmin=0, cmap='hot')
            axs[0, 2].set_title(f'vessels depth_max={str(d1+200)}')
            axs[0, 2].xaxis.set_visible(False)
            axs[0, 2].yaxis.set_visible(False)
            axs[1, 0].imshow(lymph_arr[:, :, 0], cmap='copper')
            axs[1, 0].set_title(f'lymph depth_max={str(d0)}')
            axs[1, 0].xaxis.set_visible(False)
            axs[1, 0].yaxis.set_visible(False)
            axs[1, 1].imshow(lymph_arr[:, :, 1], cmap='copper')
            axs[1, 1].set_title(f'lymph depth_max={str(d1)}')
            axs[1, 1].xaxis.set_visible(False)
            axs[1, 1].yaxis.set_visible(False)
            axs[1, 2].imshow(lymph_arr[:, :, 2], cmap='copper')
            axs[1, 2].set_title(f'lymph depth_max={str(d1+200)}')
            axs[1, 2].xaxis.set_visible(False)
            axs[1, 2].yaxis.set_visible(False)

            for ax in axs.flat:
                ax.label_outer()

            plt.savefig(f'{directory}/{name}_vessels_figure.png')

            plt.close()

            fig, axs = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw={
                                    'wspace': .01, 'hspace': -.3})

            fig.suptitle(name)

            axs[0].imshow(vess_mip[:, :, 0], vmin=.5, cmap='hot')  # , vmax=.05,
            axs[0].set_title(f'vessels mip depth_max={str(d0)}')
            axs[0].xaxis.set_visible(False)
            axs[0].yaxis.set_visible(False)
            axs[1].imshow(vess_mip[:, :, 1], vmin=.5, cmap='hot')
            axs[1].set_title(f'vessels mip depth_max={str(d1)}')
            axs[1].xaxis.set_visible(False)
            axs[1].yaxis.set_visible(False)
            axs[2].imshow(vess_mip[:, :, 2], vmin=.5, cmap='hot')
            axs[2].set_title(f'vessels mip depth_max={str(d1+200)}')
            axs[2].xaxis.set_visible(False)
            axs[2].yaxis.set_visible(False)

            for ax in axs.flat:
                ax.label_outer()

            plt.savefig(f'{directory}/{name}_vessels_mip_figure.png')

            plt.close()

            fig, axs = plt.subplots(2, 3, figsize=(10, 10), gridspec_kw={
                                    'wspace': .01, 'hspace': -.3})

            fig.suptitle(name)

            axs[0, 0].imshow(co_b_scans[0][:-50, :], vmin=.0, cmap='jet')  # , vmax=.05,
            axs[0, 0].set_title('att co B scan 100')
            axs[0, 0].xaxis.set_visible(False)
            axs[0, 0].yaxis.set_visible(False)
            axs[0, 1].imshow(co_b_scans[1][:-50, :], vmin=.0, cmap='jet')  # , vmax=.05,
            axs[0, 1].set_title('att co B scan 250')
            axs[0, 1].xaxis.set_visible(False)
            axs[0, 1].yaxis.set_visible(False)
            axs[0, 2].imshow(co_b_scans[2][:-50, :], vmin=.0, cmap='jet')  # , vmax=.05,
            axs[0, 2].set_title('att co B scan 400')
            axs[0, 2].xaxis.set_visible(False)
            axs[0, 2].yaxis.set_visible(False)

            axs[1, 0].imshow(cross_b_scans[0][:-50, :], vmin=.0, cmap='jet')  # , vmax=.05,
            axs[1, 0].set_title('att cross B scan 100')
            axs[1, 0].xaxis.set_visible(False)
            axs[1, 0].yaxis.set_visible(False)
            axs[1, 1].imshow(cross_b_scans[1][:-50, :], vmin=.0, cmap='jet')  # , vmax=.05,
            axs[1, 1].set_title('att cross B scan 250')
            axs[1, 1].xaxis.set_visible(False)
            axs[1, 1].yaxis.set_visible(False)
            axs[1, 2].imshow(cross_b_scans[2][:-50, :], vmin=.0, cmap='jet')  # , vmax=.05,
            axs[1, 2].set_title('att cross B scan 400')
            axs[1, 2].xaxis.set_visible(False)
            axs[1, 2].yaxis.set_visible(False)

            for ax in axs.flat:
                ax.label_outer()

            plt.savefig(f'{directory}/{name}_coeff_b_scans.png')

            plt.close()

            fig, axs = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw={
                                    'wspace': .1, 'hspace': .2})
            fig.suptitle(name)
            pan00 = axs[0, 0].imshow(att_co_arr[:, :, 0], vmin=0, cmap='jet')
            plt.colorbar(pan00, ax=axs[0, 0], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[0, 0].set_title('att_co'+' depth_max='+np.str(d0))
            axs[0, 0].xaxis.set_visible(False)
            axs[0, 0].yaxis.set_visible(False)
            pan01 = axs[0, 1].imshow(att_co_arr[:, :, 1], vmin=0, cmap='jet')
            plt.colorbar(pan01, ax=axs[0, 1], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[0, 1].set_title('att_co'+' depth_max='+np.str(d1))
            axs[0, 1].xaxis.set_visible(False)
            axs[0, 1].yaxis.set_visible(False)
            pan02 = axs[0, 2].imshow(att_co_arr[:, :, 2], vmin=0, cmap='jet')
            plt.colorbar(pan02, ax=axs[0, 2], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[0, 2].set_title('att_co'+' depth_max='+np.str(d1+200))
            axs[0, 2].xaxis.set_visible(False)
            axs[0, 2].yaxis.set_visible(False)
            pan10 = axs[1, 0].imshow(att_cross_arr[:, :, 0], cmap='jet')
            plt.colorbar(pan10, ax=axs[1, 0], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[1, 0].set_title('att_cross'+' depth_max='+np.str(d0))
            axs[1, 0].xaxis.set_visible(False)
            axs[1, 0].yaxis.set_visible(False)
            pan11 = axs[1, 1].imshow(att_cross_arr[:, :, 1], cmap='jet')
            plt.colorbar(pan11, ax=axs[1, 1], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[1, 1].set_title('att_cross'+' depth_max='+np.str(d1))
            axs[1, 1].xaxis.set_visible(False)
            axs[1, 1].yaxis.set_visible(False)
            pan12 = axs[1, 2].imshow(att_cross_arr[:, :, 2], cmap='jet')
            plt.colorbar(pan12, ax=axs[1, 2], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[1, 2].set_title('att_cross'+' depth_max='+np.str(d1+200))
            axs[1, 2].xaxis.set_visible(False)
            axs[1, 2].yaxis.set_visible(False)

            pan20 = axs[2, 0].imshow(att_co_arr[:, :, 0]-att_cross_arr[:, :, 0], cmap='jet')
            plt.colorbar(pan20, ax=axs[2, 0], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[2, 0].set_title('att_diff'+' depth_max='+np.str(d0))
            axs[2, 0].xaxis.set_visible(False)
            axs[2, 0].yaxis.set_visible(False)
            pan21 = axs[2, 1].imshow(att_co_arr[:, :, 1]-att_cross_arr[:, :, 1], cmap='jet')
            plt.colorbar(pan21, ax=axs[2, 1], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[2, 1].set_title('att_co'+' depth_max='+np.str(d1))
            axs[2, 1].xaxis.set_visible(False)
            axs[2, 1].yaxis.set_visible(False)
            pan22 = axs[2, 2].imshow(att_co_arr[:, :, 2]-att_cross_arr[:, :, 2], cmap='jet')
            plt.colorbar(pan22, ax=axs[2, 2], use_gridspec=True, fraction=0.046, pad=0.04)
            axs[2, 2].set_title('att_diff'+' depth_max='+np.str(d1+200))
            axs[2, 2].xaxis.set_visible(False)
            axs[2, 2].yaxis.set_visible(False)

            for ax in fig.get_axes():
                ax.label_outer()

            plt.savefig(directory+'/'+name+'_coefficients_figure.png')

            plt.close()

            for n in range(3):

                plt.imsave(directory+'/'+name+'_blood_vessels_' +
                           np.str(d_tuple[n+1])+'.png', vess_arr[:, :, n], vmin=0, cmap='hot')
                plt.imsave(directory+'/'+name+'_lymph_vessels_' +
                           np.str(d_tuple[n+1])+'.png', lymph_arr[:, :, n], vmin=0, cmap='pink')
                plt.imsave(directory+'/'+name+'_att_co' +
                           np.str(d_tuple[n+1])+'.png', att_co_arr[:, :, n], vmin=0, cmap='jet')
                plt.imsave(directory+'/'+name+'_att_cross' +
                           np.str(d_tuple[n+1])+'.png', att_cross_arr[:, :, n], vmin=0, cmap='jet')
                plt.imsave(directory+'/'+name+'_att_diff' +
                           np.str(d_tuple[n+1])+'.png', att_co_arr[:, :, n]-att_cross_arr[:, :, n], cmap='jet')

            vess_characteristics = np.append(vess_characteristics, vess_characteristics_[
                                             :, :, np.newaxis], axis=-1)
            lymph_characteristics = np.append(lymph_characteristics, lymph_characteristics_[
                                              :, :, np.newaxis], axis=-1)
            coeff_characteristics = np.append(coeff_characteristics, coeff_characteristics_[
                                              :, :, np.newaxis], axis=-1)

        except FileNotFoundError:  # (FileNotFoundError, ValueError) as e:
            print('not found')
            pass

    if len(names) > 0:
        save_xls(save_xls_path, vess_characteristics[:, :, 1:], lymph_characteristics[:, :, 1:],
                 coeff_characteristics[:, :, 1:], np.array(names))
    else:
        pass

    return (vess_characteristics[:, :, :], lymph_characteristics[:, :, :], coeff_characteristics[:, :, :], np.array(names))


def save_xls(file_path, vess_characteristics, lymph_characteristics, coeff_characteristics, names):

    print('SHAPES', names.shape, vess_characteristics.shape,
          lymph_characteristics.shape, coeff_characteristics.shape)
    #names = names[:, np.newaxis]
    title = ('name', 'density_1', 'density_2', 'density_3',
             'density_4', 'density>=5', 'mean_density')
    # 'mean_density_1', 'mean_density_2', 'mean_density_3', 'mean_density_4', 'mean_density>=5'))
    #title = title[np.newaxis, :]

    title_coeff = ('name', 'mean_co', 'std_co', 'mean_cross',
                   'std_cross', 'mean_diff', 'std_diff')
    #title_coeff = title_coeff[np.newaxis, :]

    '''
    book = xlwt.Workbook()

    sheet1 = book.add_sheet('vess_depth0')
    sheet2 = book.add_sheet('vess_depth1')
    sheet3 = book.add_sheet('vess_depth2')

    aux_tuple0 = (sheet1, sheet2, sheet3)

    sheet4 = book.add_sheet('lymph_depth0')
    sheet5 = book.add_sheet('lymph_depth1')
    sheet6 = book.add_sheet('lymph_depth2')

    aux_tuple1 = (sheet4, sheet5, sheet6)

    sheet7 = book.add_sheet('coeff_depth0')
    sheet8 = book.add_sheet('coeff_depth1')
    sheet9 = book.add_sheet('coeff_depth2')

    aux_tuple2 = (sheet7, sheet8, sheet9)
    '''

    aux_tuple0 = ('vess_depth0', 'vess_depth1', 'vess_depth2')
    aux_tuple1 = ('lymph_tuple0', 'lymph_tuple1', 'lymph_tuple2')
    aux_tuple2 = ('coeff_depth0', 'coeff_depth1', 'coeff_depth2')

    with pd.ExcelWriter(file_path) as writer:

        for n in range(3):

            #aux = np.append(names, vess_characteristics[n, :, :].T, axis=1)
            #aux = np.append(title, aux, axis=0)

            print('iter', (vess_characteristics[n, :, :][0, :]).shape)

            aux_dict = {}
            aux_dict['names'] = names
            for i, ttl in enumerate(title[1:]):
                print(ttl, vess_characteristics[n, i, :].shape)
                aux_dict[ttl] = (vess_characteristics[n, i, :])
            aux_df = pd.DataFrame(aux_dict)

            print(aux_df)
            aux_df.to_excel(writer, sheet_name=aux_tuple0[n])

            aux_dict = {}
            aux_dict['names'] = names
            for i, ttl in enumerate(title[1:]):
                aux_dict[ttl] = (lymph_characteristics[n, i, :])
            aux_df = pd.DataFrame(aux_dict)
            aux_df.to_excel(writer, sheet_name=aux_tuple1[n])

            aux_dict = {}
            aux_dict['names'] = names
            for i, ttl in enumerate(title_coeff[1:]):
                aux_dict[ttl] = (coeff_characteristics[n, i, :])
            aux_df = pd.DataFrame(aux_dict)
            aux_df.to_excel(writer, sheet_name=aux_tuple2[n])

        '''
        for i in range(aux.shape[0]):
            for j in range(aux.shape[1]):
                aux_tuple0[n].write(i, j, aux[i, j])

        aux = np.append(names, lymph_characteristics[n, :, :].T, axis=1)
        aux = np.append(title, aux, axis=0)

        for i in range(aux.shape[0]):
            for j in range(aux.shape[1]):
                aux_tuple1[n].write(i, j, aux[i, j])

        aux = np.append(names, coeff_characteristics[n, :, :].T, axis=1)
        aux = np.append(title_coeff, aux, axis=0)

        for i in range(aux.shape[0]):
            for j in range(aux.shape[1]):
                aux_tuple2[n].write(i, j, aux[i, j])
        '''

    # book.save(file_path)

    return ()


def thinning_aux(bnr_distr, N=7, tp=0):

    result = np.copy(bnr_distr)

    aux = np.copy(bnr_distr)

    if len(bnr_distr.shape) == 2:
        if tp == 0:
            kernel = np.ones((2, 2))
        else:
            kernel = morphology.disk(1)

    else:
        if tp == 0:
            kernel = np.ones((2, 2, 2))
        else:
            kernel = morphology.ball(1)

    for n in range(N):

        aux = morphology.binary_erosion(aux, kernel)

        if np.sum(aux) == 0:
            break

        result = result+(aux+.0)

    return (result)


def get_lymph_only(
    filename
):

    array = br.import_SU_bin(filename, (1024, 256, 256), np.complex64)
    co = br.get_array_tuple(filename)

    noise = np.mean(np.abs(array[40:-40, -80:-60, 40:-40][::2, :, :])**2)

    Nwind = 48

    SNR_att = signal.fftconvolve(((np.abs(array[::2, :210, :])**2-noise)**1)/(noise+0.)
                                 ** 1, np.ones((Nwind, Nwind, Nwind)), 'same')/Nwind/Nwind/Nwind

    array = signal.fftconvolve(np.abs(array[::2, :210, :])**2, np.ones((5, 1, 5)), 'same')/5/1/5
    att_co, snr = op.get_depth_resolved_attenuation(np.abs(array), Nwind=32, return_snr=True)
    att_co = att_co/.01

    #Nwind = 48
    # noise = 0.  # np.mean(np.abs(att_co[att_co<2.]))
    # noise = np.percentile(np.abs(att_co),10) #np.mean(np.abs(att_co)[20:-20, 200:220, 40:-40])
    # SNR_att = signal.fftconvolve((np.abs(att_co-noise)**1)/(noise+1.)**1,
    #                             np.ones((Nwind, Nwind, Nwind)), 'same')/Nwind/Nwind/Nwind

    rolled, supp = op.image_roll_3d((co, att_co, SNR_att), 200)

    lymph_vol = (rolled[-1] > 4.)*(rolled[1] < .4)

    lymph_map = np.sum(lymph_vol[:, 10:-25], axis=1)

    lymph_vol = 1.*(rolled[-1] > 4.)*(rolled[1] < .4)+1.*(rolled[-1] > 4.)

    return (lymph_map, lymph_vol, rolled[1])  # , rolled)


def predict_in_folder(
    directory,
    save_directory,
    save_npy_directory
):

    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    if not os.path.isdir(save_npy_directory):
        os.mkdir(save_npy_directory)

    files = os.listdir(directory)

    for i in range(len(files)):

        if not os.path.isdir(directory+'/'+files[i]):

            print(i, files[i])

            try:

                lymph_map, lymph_vol, att = get_lymph_only(directory+'/'+files[i])

                plt.figure()
                plt.imshow(lymph_map[::2, :], cmap='hot')
                plt.axis('off')
                plt.savefig(save_directory+'/' +
                            os.path.splitext(files[i])[0]+'.png', bbox_inches='tight')
                plt.close()

                np.save(save_npy_directory+'/' +
                        os.path.splitext(files[i])[0]+'.npy', lymph_vol)

            except (ValueError):
                print('pass')
                pass

        else:
            pass

    return ()
