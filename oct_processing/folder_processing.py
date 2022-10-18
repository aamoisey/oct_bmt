import numpy as np
# import re
import os
# import xlwt
from scipy import signal, stats, spatial, special
from scipy.ndimage import filters as sfilters
from skimage import morphology, measure, filters
from skimage.restoration import inpaint
from sklearn import neighbors
import pandas as pd
import matplotlib.pyplot as plt

from importlib import reload

from oct_processing import oct_processing as oct_p

reload(oct_p)


def analyze_folder(
    foldername,
    depth_xls,
    save_xls_path,
    threshold=.1,
    min_vess_size=100,
    coeff_smooth_size=3,
    calculate_vessels=True
):

    df = pd.read_excel(depth_xls)

    names = ()

    coeff_characteristics = np.zeros((3, 6, 1))

    if calculate_vessels:
        vess_characteristics = np.zeros((3, 6, 1))
        lymph_characteristics = np.zeros((3, 6, 1))

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

            if calculate_vessels:
                att_co_arr, att_cross_arr, lymph_arr, vess_arr, vess_mip, vess_characteristics_, lymph_characteristics_, coeff_characteristics_, co_b_scans, cross_b_scans = analyze_file(
                    filename=filename,
                    depth_tuple=d_tuple,
                    coeff_smooth_size=coeff_smooth_size,
                    threshold=threshold,
                    min_vess_size=min_vess_size,
                    calculate_vessels=calculate_vessels)

            else:
                att_co_arr, att_cross_arr, coeff_characteristics_, co_b_scans, cross_b_scans = analyze_file(
                    filename=filename,
                    depth_tuple=d_tuple,
                    coeff_smooth_size=coeff_smooth_size,
                    threshold=threshold,
                    min_vess_size=min_vess_size,
                    calculate_vessels=calculate_vessels)

            names = names+(name,)

            directory = f'{foldername}/{name}'
            if os.path.isdir(directory) == False:
                os.mkdir(directory)

            if calculate_vessels:
                fig_vess, axs_vess = plt.subplots(2, 3, figsize=(10, 10), gridspec_kw={
                    'wspace': .01, 'hspace': -.3})
                fig_mip, axs_mip = plt.subplots(1, 3, figsize=(10, 10), gridspec_kw={
                    'wspace': .01, 'hspace': -.3})
                fig_vess.suptitle(name)
                fig_mip.suptitle(name)

            fig_coeff, axs_coeff = plt.subplots(2, 3, figsize=(10, 10), gridspec_kw={
                'wspace': .01, 'hspace': -.3})
            fig_map, axs_map = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw={
                'wspace': .1, 'hspace': .2})
            fig_coeff.suptitle(name)
            fig_map.suptitle(name)

            y_tuple = [100, 250, 400]

            for n in range(3):

                if calculate_vessels:
                    axs_vess[0, n].imshow(vess_arr[:, :, n], vmin=0, cmap='hot')
                    axs_vess[0, n].set_title(f'vess depth_max={str(d_tuple[n+1])}')
                    axs_vess[0, n].xaxis.set_visible(False)
                    axs_vess[0, n].xaxis.set_visible(False)

                    axs_vess[1, n].imshow(lymph_arr[:, :, n], cmap='copper')
                    axs_vess[1, n].set_title(f'lymph depth_max={str(d_tuple[n+1])}')
                    axs_vess[1, n].xaxis.set_visible(False)
                    axs_vess[1, n].xaxis.set_visible(False)

                    axs_mip[n].imshow(vess_mip[:, :, n], vmin=.5, cmap='hot')
                    axs_mip[n].set_title(f'vessels mip depth_max={str(d_tuple[n+1])}')
                    axs_mip[n].xaxis.set_visible(False)
                    axs_mip[n].xaxis.set_visible(False)

                axs_coeff[0, n].imshow(co_b_scans[n][:-50, :], vmin=0, cmap='jet')
                axs_coeff[0, n].set_title(f'att co B scan {y_tuple[n]}')
                axs_coeff[0, n].xaxis.set_visible(False)
                axs_coeff[0, n].xaxis.set_visible(False)

                axs_coeff[1, n].imshow(cross_b_scans[n][:-50, :], cmap='jet')
                axs_coeff[1, n].set_title(f'att cross B scan {y_tuple[n]}')
                axs_coeff[1, n].xaxis.set_visible(False)
                axs_coeff[1, n].xaxis.set_visible(False)

                pan = axs_map[0, n].imshow(att_co_arr[:, :, n], vmin=0, cmap='jet')
                plt.colorbar(pan, ax=axs_map[0, n], use_gridspec=True, fraction=.046, pad=.04)
                axs_map[0, n].set_title(f'att_co depth_max={str(d_tuple[n+1])}')
                axs_map[0, n].xaxis.set_visible(False)
                axs_map[0, n].yaxis.set_visible(False)

                pan = axs_map[1, n].imshow(att_cross_arr[:, :, n], cmap='jet')
                plt.colorbar(pan, ax=axs_map[1, n], use_gridspec=True, fraction=.046, pad=.04)
                axs_map[1, n].set_title(f'att_cross depth_max={str(d_tuple[n+1])}')
                axs_map[1, n].xaxis.set_visible(False)
                axs_map[1, n].yaxis.set_visible(False)

                pan = axs_map[2, n].imshow(att_co_arr[:, :, n]-att_cross_arr[:, :, n], cmap='jet')
                plt.colorbar(pan, ax=axs_map[2, n], use_gridspec=True, fraction=.046, pad=.04)
                axs_map[2, n].set_title(f'att_diff depth_max={str(d_tuple[n+1])}')
                axs_map[2, n].xaxis.set_visible(False)
                axs_map[2, n].yaxis.set_visible(False)

            if calculate_vessels:
                for ax in axs_vess.flat:
                    ax.label_outer()
                for ax in axs_mip.flat:
                    ax.label_outer()

            for ax in axs_coeff.flat:
                ax.label_outer()
            for ax in axs_map.flat:
                ax.label_outer()

            if calculate_vessels:
                fig_vess.savefig(f'{directory}/{name}_vessels_figure.png')
                plt.close(fig_vess)
                fig_mip.savefig(f'{directory}/{name}_vessels_mip_figure.png')
                plt.close(fig_mip)

            fig_coeff.savefig(f'{directory}/{name}_coeff_b_scans.png')
            plt.close(fig_coeff)
            fig_map.savefig(f'{directory}/{name}_coeff_map.png')
            plt.close(fig_map)

            for n in range(3):
                if calculate_vessels:
                    plt.imsave(
                        f'{directory}/{name}_blood_vessels_{np.str(d_tuple[n+1])}.png', vess_arr[:, :, n], vmin=0, cmap='hot')
                    plt.imsave(
                        f'{directory}{name}_lymph_vessels_{np.str(d_tuple[n+1])}.png', lymph_arr[:, :, n], vmin=0, cmap='pink')
                plt.imsave(
                    f'{directory}/{name}_att_co_{np.str(d_tuple[n+1])}.png', att_co_arr[:, :, n], vmin=0, cmap='jet')
                plt.imsave(
                    f'{directory}/{name}_att_cross_{np.str(d_tuple[n+1])}.png', att_cross_arr[:, :, n], vmin=0, cmap='jet')
                plt.imsave(
                    f'{directory}/{name}_att_diff_{np.str(d_tuple[n+1])}.png', att_co_arr[:, :, n]-att_cross_arr[:, :, n], cmap='jet')

            if calculate_vessels:
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
        if calculate_vessels:
            save_xls(save_xls_path,
                     (vess_characteristics[:, :, 1:],
                      lymph_characteristics[:, :, 1:],
                      coeff_characteristics[:, :, 1:]),
                     np.array(names))
        else:
            save_xls(save_xls_path,
                     (coeff_characteristics[:, :, 1:],),
                     np.array(names))
    else:
        pass

    '''
    if calculate_vessels:
        return (vess_characteristics[:, :, :],
                lymph_characteristics[:, :, :],
                coeff_characteristics[:, :, :],
                np.array(names))
    else:
        return (coeff_characteristics[:, :, :],
                np.array(names))
    '''

    return ()


def analyze_file(
    filename,
    depth_tuple,
    threshold=.1,
    min_vess_size=100,
    coeff_smooth_size=3,
    calculate_vessels=True
):

    if calculate_vessels:
        att_co, att_cross, vess, lymph, att_co_unshifted, att_cross_unshifted = get_3d_distributions(filename,
                                                                                                     coeff_smooth_size=coeff_smooth_size,
                                                                                                     calculate_vessels=calculate_vessels)
        att_co_arr, att_cross_arr, lymph_arr, vess_arr, vess_mip, co_b_scans, cross_b_scans = get_maps(
            (att_co, att_cross, lymph, vess, att_co_unshifted, att_cross_unshifted),
            depth_tuple,
            threshold,
            min_vess_size)

        lymph = signal.resample(1.*lymph[:, :, 30:-30], lymph.shape[-1], axis=-1)

        lymph_skel_3d = get_thickness_volume(lymph[::2, :, :] > .01, N=17)

        vess_characteristics = np.zeros((len(depth_tuple)-1, 6))
        lymph_characteristics = np.zeros((len(depth_tuple)-1, 6))

    else:
        att_co, att_cross, att_co_unshifted, att_cross_unshifted = get_3d_distributions(filename,
                                                                                        coeff_smooth_size=coeff_smooth_size,
                                                                                        calculate_vessels=calculate_vessels)
        att_co_arr, att_cross_arr, co_b_scans, cross_b_scans = get_maps(
            (att_co, att_cross, att_co_unshifted, att_cross_unshifted),
            depth_tuple,
            threshold)  # , min_vess_size)

    coeff_characteristics = np.zeros((len(depth_tuple)-1, 6))

    for i in range(len(depth_tuple)-1):

        if calculate_vessels:
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

    if calculate_vessels:
        vess_characteristics[:, :5] = vess_characteristics[:, :5]/np.prod(vess_arr[:, :, 0].shape)
        lymph_characteristics[:, :5] = lymph_characteristics[:, :5]/np.prod(vess_arr[:, :, 0].shape)

    if calculate_vessels:
        return (att_co_arr,
                att_cross_arr,
                lymph_arr,
                vess_arr,
                vess_mip,
                vess_characteristics,
                lymph_characteristics,
                coeff_characteristics,
                co_b_scans,
                cross_b_scans)
    else:
        return (att_co_arr,
                att_cross_arr,
                coeff_characteristics,
                co_b_scans,
                cross_b_scans)


def save_xls(file_path,
             characteristics_tuple,
             names):

    if len(characteristics_tuple) == 3:
        vess_characteristics = characteristics_tuple[0]
        lymph_characteristics = characteristics_tuple[1]
        coeff_characteristics = characteristics_tuple[2]
    else:
        coeff_characteristics = characteristics_tuple[0]

    # print('SHAPES', names.shape, vess_characteristics.shape,
    #      lymph_characteristics.shape, coeff_characteristics.shape)

    # names = names[:, np.newaxis]

    if len(characteristics_tuple) == 3:
        title = ('name', 'density_1', 'density_2', 'density_3',
                 'density_4', 'density>=5', 'mean_density')

    title_coeff = ('name', 'mean_co', 'std_co', 'mean_cross',
                   'std_cross', 'mean_diff', 'std_diff')
    # title_coeff = title_coeff[np.newaxis, :]

    if len(characteristics_tuple) == 3:
        aux_tuple0 = ('vess_depth0', 'vess_depth1', 'vess_depth2')
        aux_tuple1 = ('lymph_tuple0', 'lymph_tuple1', 'lymph_tuple2')
    aux_tuple2 = ('coeff_depth0', 'coeff_depth1', 'coeff_depth2')

    with pd.ExcelWriter(file_path) as writer:

        for n in range(3):

            # aux = np.append(names, vess_characteristics[n, :, :].T, axis=1)
            # aux = np.append(title, aux, axis=0)

            if len(characteristics_tuple) == 3:
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

    return ()


def get_3d_distributions(
        filename,
        coeff_smooth_size=3,
        calculate_vessels=True
):

    if calculate_vessels:
        array, vess = oct_p.get_vessels_from_filepath(filename)
    else:
        array = oct_p.load_oct_data(filename,
                                    shape=(1024, 256, 256),
                                    dtype='complex64')

    co = get_normalized_array(array)

    array_ = (signal.fftconvolve(np.abs(array[1::2, :, :])**2,
                                 np.ones((coeff_smooth_size, coeff_smooth_size, coeff_smooth_size)), 'same') /
              coeff_smooth_size/coeff_smooth_size/coeff_smooth_size)
    # array_ = signal.fftconvolve(np.abs(array)**2, np.ones((5, 5, 5)), 'same')/5/5/5
    att_cross = oct_p.get_depth_resolved_attenuation(np.abs(array_), Nwind=32)
    att_cross = att_cross/.01

    array = (signal.fftconvolve(np.abs(array[::2, :, :])**2,
                                np.ones((coeff_smooth_size, coeff_smooth_size, coeff_smooth_size)), 'same') /
             coeff_smooth_size/coeff_smooth_size/coeff_smooth_size)
    # array = signal.fftconvolve(np.abs(array)**2, np.ones((5, 5, 5)), 'same')/5/5/5
    att_co, snr = oct_p.get_depth_resolved_attenuation(np.abs(array), Nwind=32, return_snr=True)
    att_co = att_co/.01

    if calculate_vessels:
        dmtrt = 24  # 16
        bg = signal.fftconvolve(att_co, np.ones((dmtrt, dmtrt, dmtrt)), 'same')/dmtrt/dmtrt/dmtrt
        aux = (att_co-bg)**2
        aux[aux > 48] = 48  # 32
        var = signal.fftconvolve(aux, np.ones((dmtrt, dmtrt, dmtrt)), 'same')/dmtrt/dmtrt/dmtrt

        rolled, supp = image_roll_3d(
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

        return (rolled[1],
                rolled[2],
                rolled[3],
                lymph_clean,
                att_co,
                att_cross)

    else:

        rolled, supp = image_roll_3d((co, att_co, att_cross), 210)

        return (rolled[1],
                rolled[2],
                att_co,
                att_cross)


def get_maps(distr_tuple,
             depth_tuple,
             threshold=.1,
             min_vess_size=100):

    if len(distr_tuple) == 6:
        att_co = distr_tuple[0]
        att_cross = distr_tuple[1]
        lymph = distr_tuple[2]
        vess = distr_tuple[3]
        att_co_unshifted = distr_tuple[4]
        att_cross_unshifted = distr_tuple[5]
    else:
        att_co = distr_tuple[0]
        att_cross = distr_tuple[1]
        att_co_unshifted = distr_tuple[2]
        att_cross_unshifted = distr_tuple[3]

    att_co_arr = np.zeros((np.int(att_co.shape[0]/2), att_co.shape[-1], len(depth_tuple)-1))
    att_cross_arr = np.zeros(
        (np.int(att_cross.shape[0]/2), att_cross.shape[-1], len(depth_tuple)-1))

    if len(distr_tuple) == 6:
        lymph_arr = np.zeros((np.int(lymph.shape[0]/2), lymph.shape[-1], len(depth_tuple)-1))
        vess_arr = np.zeros((np.int(vess.shape[0]/2), vess.shape[-1], len(depth_tuple)-1))
        vess_mip_arr = np.zeros((np.int(vess.shape[0]/2), vess.shape[-1], len(depth_tuple)-1))

    for i in range(len(depth_tuple)-1):

        d = (depth_tuple[i+1]+depth_tuple[i])/2.
        N = np.int(d/6.)
        h = 8
        Nmin = max(0, N-h)
        Nmax = min(N+h, att_co.shape[0])

        if len(distr_tuple) == 6:
            lymph_map = np.mean(lymph[::2, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :], axis=1)  # aux-dil_bg
            lymph_map = signal.resample(lymph_map[:, 30:-30], lymph_map.shape[1], axis=1)

            vess_mip_arr[:, :, i] = np.max(
                vess[::2, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :], axis=1)
            aux = vess_mip_arr[:, :, i] - \
                np.median(vess[::2, np.int(depth_tuple[i]/6):np.int(depth_tuple[i+1]/6), :], axis=1)

            aux_ = filters.gaussian(aux, 1)

            print('max_aux', np.max(aux), np.median(aux))

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
            # vess_map = np.zeros(vess_map.shape)

            vess_mask = morphology.remove_small_objects(vess_map > threshold, min_vess_size)
            vess_map = vess_map*vess_mask

            '''
            mask = np.sum(vess_map, axis=1) > 20
            mask = mask[:, np.newaxis]
            mask = np.ones(vess_map.shape)*mask
            mask = morphology.binary_dilation(mask > 0, np.ones((1, 1)))
            '''

        att_co_map = np.mean(att_co[::2, Nmin:Nmax, :], axis=1)
        att_cross_map = np.mean(att_cross[::2, Nmin:Nmax, :], axis=1)
        att_diff_map = np.mean(att_co[::2, Nmin:Nmax, :]-att_cross[::2, Nmin:Nmax, :], axis=1)

        att_co_map = signal.resample(att_co_map[:, 30:-30], att_co_map.shape[1], axis=1)
        att_cross_map = signal.resample(att_cross_map[:, 30:-30], att_cross_map.shape[1], axis=1)
        att_diff_map = signal.resample(att_diff_map[:, 30:-30], att_diff_map.shape[1], axis=1)

        att_co_arr[:, :, i] = att_co_map
        att_cross_arr[:, :, i] = att_cross_map

        if len(distr_tuple) == 6:
            lymph_arr[:, :, i] = lymph_map
            vess_arr[:, :, i] = vess_map

    if len(distr_tuple) == 6:
        return (att_co_arr,
                att_cross_arr,
                lymph_arr,
                vess_arr,
                vess_mip_arr,
                (att_co_unshifted[200, :, :],
                 att_co_unshifted[500, :, :],
                 att_co_unshifted[-200, :, :]),
                (att_cross_unshifted[200, :, :],
                 att_cross_unshifted[500, :, :],
                 att_cross_unshifted[-200, :, :]))
    else:
        return (att_co_arr,
                att_cross_arr,
                (att_co_unshifted[200, :, :],
                 att_co_unshifted[500, :, :],
                 att_co_unshifted[-200, :, :]),
                (att_cross_unshifted[200, :, :],
                 att_cross_unshifted[500, :, :],
                 att_cross_unshifted[-200, :, :]))


def get_normalized_array(array):

    co = np.abs(array[::2, :, :])

    aux = signal.fftconvolve(co, np.ones((7, 1, 1)), 'same')
    co_ = co/(aux**2+5000**2)**.5

    return (co_)  # ,co,cross)


def image_roll_3d(
    array_tuple,
    N=100,
    n=15,
    d=8,
    D=16
):

    array = array_tuple[0]

    x = np.linspace(1, 0, array.shape[1])

    supp = np.zeros((np.int(array.shape[0]/d), array.shape[1], array.shape[2]))

    for i in range(np.int(array.shape[0]/d)):

        #        aux = array[i*d,:,:]*(glass_[:,i*d]>0)[:,np.newaxis]
        aux = array[i*d, :, :]
        supp[i, :, :], _ = support(aux, n)  # array[i*d,:,:],n)

    argmax = np.argmax(x[np.newaxis, :, np.newaxis]*supp, axis=1)  # +10
#    argmax = np.argmax(array_tuple[0][::d,:,:]*supp,axis=1)

    indxs = interpolate(argmax, (array.shape[0], array.shape[2]), D)

    L = len(array_tuple)

    aux = np.zeros((array_tuple[0].shape[0], N, array_tuple[0].shape[2]))

    for i in range(array_tuple[0].shape[0]):
        for j in range(array_tuple[0].shape[2]):
            aux[i, :, j] = np.roll(array_tuple[0][i, :, j], np.int(-indxs[i, j]))[:N]

    result = (aux,)

    for l in range(1, L):
        aux = np.zeros((array_tuple[0].shape[0], N, array_tuple[0].shape[2]))
        for i in range(array_tuple[l].shape[0]):
            for j in range(array_tuple[l].shape[2]):
                aux[i, :, j] = np.roll(array_tuple[l][i, :, j], np.int(-indxs[i, j]))[:N]

        result += (aux,)

    return (result, supp)


def support(
        image,
        n=15
):

    aux = sfilters.median_filter(image, footprint=np.ones((n, n)))

    th = filters.threshold_otsu(aux[aux > 0])

    aux = morphology.remove_small_objects(aux > th/1., 20)
    aux = 1.*aux

    x = np.arange(image.shape[0])
    aux *= x[:, np.newaxis]

    cm = np.percentile(aux, 95, 1)

#    print(x.shape,cm.shape)

    label = measure.label(aux > 0, background=0)
#    print(np.max(label))

    label_ind = np.zeros(image.shape)
    label_ind[np.int16(cm), np.int16(x)] = 1

#    for i in range(image.shape[1]):
#        label_ind[i] = label[np.int(cm[i]),i]

    label_ind = label[label_ind > 0]

    result = np.zeros(image.shape)

    for i in np.unique(label_ind[label_ind > 0]):

        result += (label == i)

    return (result, cm)


def interpolate(image, size, N=16):

    d = np.int(size[0]/image.shape[0])

    x = np.arange(size[1])
    y = np.arange(size[0])
    X, Y = np.meshgrid(x, y)

    Y = Y/2

    mask = image < 10
    image = inpaint.inpaint_biharmonic(1.*image, 1.*mask)

    aux = np.zeros(size)
    aux[::d, :] = image

    knn = neighbors.KNeighborsRegressor(N, weights='distance')
    knn.fit(np.append(X[aux > 0][:, np.newaxis], Y[aux > 0][:, np.newaxis], axis=1), aux[aux > 0])

    interpolation = knn.predict(
        np.append(X[aux >= 0][:, np.newaxis], Y[aux >= 0][:, np.newaxis], axis=1))

    return (np.int16(interpolation.reshape(size)))


def get_thickness_volume(
    bnr_distr_3d,
    N=17
):

    bnr_distr_3d_ = morphology.binary_closing(bnr_distr_3d, morphology.ball(6))
    skeleton = morphology.skeletonize_3d(bnr_distr_3d_)
    aux = thinning_aux(bnr_distr_3d_, N)

    return (aux*(skeleton > 0))


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


def thinning_aux(
    bnr_distr,
    N=7,
    tp=0
):

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
