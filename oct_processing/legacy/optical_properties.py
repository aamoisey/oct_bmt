import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#import re
import os
#import xlwt
from sklearn import neighbors
from skimage.restoration import inpaint  # , denoise_tv_chambolle as tv
from scipy.ndimage import filters as sfilters
#from scipy import misc
from skimage import filters, morphology, measure


def import_SU_bin(filename, size=(256, 256, 256), mode='complex64'):

    F = open(filename, 'rb')

    A = F.read()
    B = np.fromstring(A, dtype=mode)
    print(B.shape)
    B = np.reshape(B, size)

    return (B)


def support(image, n=15):

    aux = sfilters.median_filter(image, footprint=np.ones((n, n)))

    th = filters.threshold_otsu(aux[aux > 0])

    aux = morphology.remove_small_objects(aux > th/1., 20)
    aux = 1.*aux

    x = np.arange(image.shape[0])
    aux *= x[:, np.newaxis]

#    cm = np.zeros(image.shape[1])

#    for i in range(image.shape[1]):
#        cm[i] = np.median(aux[:,i][aux[:,i]>0])

#    cm[np.isnan(cm)] = 0

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


def image_roll_3d(array_tuple, N=100, n=15, d=8, D=16):

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

    return (result, supp)  # argmax,indxs,supp) #close_small,close_all_smooth,close_all,indxs)


def get_array_tuple(filename):

    # this part to separate function!!!
    co = import_SU_bin(filename, (1024, 256, 256), np.complex64)
    cross = np.abs(co[1::2, :, :])
    co = np.abs(co[::2, :, :])

    aux = signal.fftconvolve(co, np.ones((7, 1, 1)), 'same')
    co_ = co/(aux**2+5000**2)**.5

#    aux = signal.fftconvolve(cross,np.ones((7,1,1)),'same')
#
#    co = co_
#    cross = cross/(aux**2+5000**2)**.5

    return (co_, co, cross)


def r_hand_(array_co, array_cross):

    if len(array_co.shape) == 1:
        result = (np.array(array_co), np.array(array_cross))

    elif len(array_co.shape) == 2:
        result = (np.array(array_co), np.array(array_cross))

    else:
        aux = np.ones((array_co.shape[0], array_co.shape[2]))
        result = (np.array((array_co.transpose((0, 2, 1))[aux > 0]).T), np.array(
            (array_cross.transpose((0, 2, 1))[aux > 0]).T))

    return (result)


############################################################################################
############################################################################################

'''
def get_depth_resolved_attenuation(array, Nwind=32):

    Nmax = array.shape[1]+1
    z = np.arange(array.shape[1])
    z = z[np.newaxis, :, np.newaxis]

    noise = np.mean(array[20:-20, -30:-20, 40:-40])
    print(noise)
    norm = np.cumsum(array[:, -1::-1, :], axis=1)[:, -1::-1, :]

    mu_est = array/norm
    noise_ = noise/norm

    #noise__ = np.cumsum(noise*np.ones(array.shape[1])[-1::-1])[-1::-1]
    #noise__ = noise__[np.newaxis, :, np.newaxis]

    H = 1.-noise_*(Nmax-z)  # noise*(Nmax-z)*(Nmax+z+1)/2./norm

    SNR = signal.fftconvolve(((mu_est-noise_)**2)/noise_**2,
                             np.ones((1, Nwind, Nwind)), 'same')/Nwind/Nwind
    SNR[SNR < 0] = 0

    attenuation = mu_est*H*SNR/(H*H*SNR+1)

    return(attenuation, H, SNR, noise_, mu_est)
'''


def get_depth_resolved_attenuation(array, Nwind=32, return_snr=False):

    Nmax = array.shape[1]+1
    z = np.arange(array.shape[1])
    z = z[np.newaxis, :, np.newaxis]

    noise = np.mean(array[20:-20, -30:-20, 40:-40])
    print(noise)
    norm = np.cumsum(array[:, -1::-1, :], axis=1)[:, -1::-1, :]

    mu_est = array/norm
    #noise_ = noise/norm

    #noise__ = np.cumsum(noise*np.ones(array.shape[1])[-1::-1])[-1::-1]
    #noise__ = noise__[np.newaxis, :, np.newaxis]

    H = 1.-noise*(Nmax-z)/norm  # noise*(Nmax-z)*(Nmax+z+1)/2./norm

    SNR = signal.fftconvolve((np.abs(array-noise)**1)/noise**1,
                             np.ones((1, Nwind, Nwind)), 'same')/Nwind/Nwind
    #SNR[SNR < 0] = 0

    attenuation = mu_est*H*SNR/(H*H*SNR+1)

    if return_snr:
        return (attenuation, SNR)
    else:
        return (attenuation)  # , H, SNR, mu_est)


def optical_coeff_inv_pixels(filename, Nmax=100, Nmin=40, variant=2):

    x = np.arange(Nmax-Nmin)

    #array = import_SU_bin(filename, (1024, 256, 256), np.complex64)

    # array_ = np.abs(array[1::2, :, :])**1  # -np.abs(array[1::2,:,:])**2
    # array = np.abs(array[::2, :, :])**1  # +np.abs(array[1::2,:,:])**2

    co_, array, array_ = get_array_tuple(filename)

    if variant == 1:
        array_ = np.abs(array**2-array_**2)  # +10**-10
        array = array**2+array_**2
        #array_ = array/array_
    else:
        array = np.abs(array)**2
        array_ = np.abs(array_)**2

    array = sfilters.uniform_filter(array+.0, 5)
    array_ = sfilters.uniform_filter(array_+.0, 5)

    rolled, supp = image_roll_3d((co_, array, array_), Nmax)

    y = r_hand_(rolled[1][:, Nmin:, :], rolled[2][:, Nmin:, :])

    coeff_co = np.polyfit(x, np.log(y[0]), 1)
    coeff_cross = np.polyfit(x, np.log(y[1]), 1)
    coeff_co = coeff_co[0].reshape((-1, array.shape[-1]))
    coeff_cross = coeff_cross[0].reshape((-1, array.shape[-1]))

    if variant == 1:
        att_co = -coeff_co
        att_cross_forward = -(coeff_co-coeff_cross)/2.
        return (att_co, att_cross_forward)
    else:
        att_cross = -coeff_co
        att_co = -coeff_cross
        return (att_co, att_cross)


def get_inv_mm_map(filename, Nmax=100, Nmin=40, variant=2, conversion_coefficient=.007):

    # variant==1 - forward cross-scattering
    # variant==2 - differential attenuation
    #variant==3 - depth_resolved

    if variant < 3:
        map_co, map_cross = optical_coeff_inv_pixels(filename, Nmax, Nmin, variant)
    elif variant == 3:
        co_, array, array_ = get_array_tuple(filename)
        att_co = get_depth_resolved_attenuation(np.abs(array)**2, Nwind=32)
        att_cross = get_depth_resolved_attenuation(np.abs(array_)**2, Nwind=32)

        rolled, supp = image_roll_3d((co_, att_co, att_cross), Nmax)

        map_co = np.mean(rolled[1][:, Nmin:, :], axis=1)
        map_cross = np.mean(rolled[2][:, Nmin:, :], axis=1)
    else:
        co_, array, array_ = get_array_tuple(filename)
        array = signal.fftconvolve(np.abs(array)**2, np.ones((5, 1, 5)), 'same')/5/5/5
        array_ = signal.fftconvolve(np.abs(array_)**2, np.ones((5, 1, 5)), 'same')/5/5/5
        att_co = get_depth_resolved_attenuation(array, Nwind=32)
        att_cross = get_depth_resolved_attenuation(array_, Nwind=32)

        rolled, supp = image_roll_3d((co_, att_co, att_cross), Nmax)

        map_co = np.mean(rolled[1][:, Nmin:, :], axis=1)
        map_cross = np.mean(rolled[2][:, Nmin:, :], axis=1)

    map_co = map_co/conversion_coefficient
    map_cross = map_cross/conversion_coefficient

    return (map_co, map_cross)

############################################################################################
############################################################################################


def do_in_folder(folder_name,
                 Nmax,
                 Nmin,
                 Nindent,
                 variant=2,
                 conversion_coefficient=.007,
                 vmin_co=6,
                 vmax_co=10,
                 vmin_cross=6,
                 vmax_cross=10,
                 vmin_diff=0,
                 vmax_diff=4):

    files_list = os.listdir(folder_name)
    save_npy = folder_name+'/_npy'
    save_xls = folder_name+'/_xls'
    save_map = folder_name+'/_map'

    if variant == 1:
        header = np.array(['attenuation', 'forward crosscattering'])
    else:
        header = np.array(['attenuation_co', 'attenuation_cross', 'differential attenuation'])
    #header = header[np.newaxis, :]

    for f in files_list:

        print(f)
        filename = ''.join([folder_name, '/', f])
        split_ext = os.path.splitext(f)[0]
        if os.path.isdir(filename) == 0:

            # map_co, map_cross = get_inv_mm_map(
            #    filename, Nmax, Nmin, variant, conversion_coefficient)

            try:

                map_co, map_cross = get_inv_mm_map(
                    filename, Nmax, Nmin, variant, conversion_coefficient)

                if os.path.isdir(save_npy) == 0:
                    os.makedirs(save_npy)
                if os.path.isdir(save_xls) == 0:
                    os.makedirs(save_xls)
                if os.path.isdir(save_map) == 0:
                    os.makedirs(save_map)

                if variant == 1:
                    np.save(save_npy+'/'+split_ext+'_variant_' +
                            np.str(variant)+'_map_co'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.npy', map_co)
                    np.save(save_npy+'/'+split_ext+'_variant_' +
                            np.str(variant)+'_map_cross'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.npy', map_cross)
                    data = np.append(map_co[:, Nindent:-Nindent].ravel()[::16][:, np.newaxis],
                                     map_cross[:, Nindent:-Nindent].ravel()[::16][:, np.newaxis], axis=1)
                    #data = np.append(header, data, axis=0)
                    plt.imsave(save_map+'/'+split_ext+'_variant_'+np.str(variant) +
                               '_att_co'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.png', map_co[::2, :], cmap='jet', vmin=vmin_co, vmax=vmax_co)
                    plt.imsave(save_map+'/'+split_ext+'_variant_'+np.str(variant) +
                               '_forward_cc'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.png', map_cross[::2, :], cmap='jet', vmin=vmin_cross, vmax=vmax_cross)
                else:
                    map_diff = map_co-map_cross
                    np.save(save_npy+'/'+split_ext+'_variant_' +
                            np.str(variant)+'_map_co'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.npy', map_co)
                    np.save(save_npy+'/'+split_ext+'_variant_' +
                            np.str(variant)+'_map_cross'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.npy', map_cross)
                    np.save(save_npy+'/'+split_ext+'_variant_' +
                            np.str(variant)+'_map_diff'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.npy', map_diff)
                    data = np.append(map_co[:, Nindent:-Nindent].ravel()[::16][:, np.newaxis],
                                     map_cross[:, Nindent:-Nindent].ravel()[::16][:, np.newaxis], axis=1)
                    data = np.append(data, map_diff[:, Nindent:-
                                                    Nindent].ravel()[::16][:, np.newaxis], axis=1)
                    #data = np.append(header, data, axis=0)
                    plt.imsave(save_map+'/'+split_ext+'_variant_'+np.str(variant) +
                               '_att_co'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.png', map_co[::2, :], cmap='jet', vmin=vmin_co, vmax=vmax_co)
                    plt.imsave(save_map+'/'+split_ext+'_variant_'+np.str(variant) +
                               '_att_cross'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.png', map_cross[::2, :], cmap='jet', vmin=vmin_cross, vmax=vmax_cross)
                    plt.imsave(save_map+'/'+split_ext+'_variant_'+np.str(variant) +
                               '_diff_att'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.png', map_diff[::2, :], cmap='jet', vmin=vmin_diff, vmax=vmax_diff)

                write_xls(save_xls+'/'+split_ext+'_variant_' +
                          np.str(variant)+'_3d'+'_from'+np.str(Nmin)+'to'+np.str(Nmax)+'.xls', data, header)

            except IOError:
                pass
            except ValueError:
                pass
#            except np.linalg.LinAlgError:
#                pass

        else:
            do_in_folder(filename, Nmax, Nmin, Nindent, variant, conversion_coefficient)

    return ()


def do_in_directory(directory_name, Nmax, Nmin, Nindent, variant=2, conversion_coefficient=.007):

    files_list = os.listdir(directory_name)

    for f in files_list:

        if os.path.isfile(directory_name+'/'+f):
            pass
        else:
            do_in_folder(directory_name+'/'+f, Nmax, Nmin, Nindent,
                         variant=2, conversion_coefficient=.007)

    return ()  # data,names)

############################################################################################
############################################################################################


def write_xls(file_path, data, header):

    return ()


'''

    print(header[0])

    mean = np.mean(data[:, :], axis=0)
    median = np.median(data[:, :], axis=0)
    std = np.std(data[:, :], axis=0)

    if data.shape[1] == 3:
        title0 = 'mean '+header[0]
        title1 = 'mean '+header[1]
        title2 = 'mean '+header[2]
        title3 = 'median ' + header[0]
        title4 = 'median '+header[1]
        title5 = 'median '+header[2]
        title6 = 'std '+header[0]
        title7 = 'std '+header[1]
        title8 = 'std '+header[2]
        header_stats = np.array((title0, title1, title2, title3,
                                 title4, title5, title6, title7, title8))
    else:
        title0 = 'mean '+header[0]
        title1 = 'mean '+header[1]
        title2 = 'median ' + header[0]
        title3 = 'median '+header[1]
        title4 = 'std '+header[0]
        title5 = 'std '+header[1]
        header_stats = np.array((title0, title1, title2, title3, title4, title5))
    values_stats = np.append(mean, median)
    values_stats = np.append(values_stats, std)
    data_stats = np.append(header_stats[np.newaxis, :], values_stats[np.newaxis, :], axis=0)

    data = np.append(header[np.newaxis, :], data, axis=0)

    N, M = data.shape

    book = xlwt.Workbook()
#    print('book')
    sheet1 = book.add_sheet('values')
    sheet2 = book.add_sheet('stats')
#    print('sheet')

    for i in range(N):
        #        print('write data',i)
        for j in range(M):
            sheet1.write(i, j, data[i, j])

    N, M = data_stats.shape

    for i in range(N):
        #        print('write data',i)
        for j in range(M):
            sheet2.write(i, j, data_stats[i, j])

    book.save(file_path)

'''


def write_csv(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            f.write(",".join(line) + "\n")


def data_from_folder(folder_name, three_D=True):

    files_list = os.listdir(folder_name)

    n = 0

    for f in files_list:
        #    for n in range(1,N):

        #        f = files_list[n]

        #        print(f[-6:])

        if three_D == True:
            condition = (f[-6:] == '3d.npy')
        else:
            condition = (f[-9:] == 'axis0.npy')

        if condition:  # 'axis0.npy':

            print(n, f)
            filename = ''.join([folder_name, '/', f])

            data_ = np.load(filename)

            if n == 0:
                data = data_[:, :, np.newaxis]
            else:
                data = np.append(data, data_[:, :, np.newaxis], axis=-1)

            n += 1

        else:
            pass

    return (data)
