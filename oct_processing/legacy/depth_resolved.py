import numpy as np

from scipy import signal
#from skimage import morphology
from scipy.ndimage import filters as sfilters

from skimage import filters, morphology, measure
from skimage.restoration import inpaint

#from sklearn.cluster import MiniBatchKMeans
#from sklearn.decomposition import PCA
from sklearn import neighbors

#import os

#import xlwt


def import_SU_bin(filename, size=(256, 256, 256), mode='complex64'):

    F = open(filename, 'rb')

    A = F.read()
    B = np.fromstring(A, dtype=mode)
    print(B.shape)
    B = np.reshape(B, size)

    return(B)


def support(image, n=15):

    aux = sfilters.median_filter(image, footprint=np.ones((n, n)))

    th = filters.threshold_otsu(aux[aux > 0])

    aux = morphology.remove_small_objects(aux > th, 20)
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

    return(result, cm)


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

    knn = neighbors.KNeighborsRegressor(N, weights='uniform')
    knn.fit(np.append(X[aux > 0][:, np.newaxis], Y[aux > 0][:, np.newaxis], axis=1), aux[aux > 0])

    interpolation = knn.predict(
        np.append(X[aux >= 0][:, np.newaxis], Y[aux >= 0][:, np.newaxis], axis=1))

    return(np.int16(interpolation.reshape(size)))


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

    return(result, supp)  # argmax,indxs,supp) #close_small,close_all_smooth,close_all,indxs)


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

    return(co_, co, cross)


def get_depth_resolved_coefficients(abs_array, reg_param=100.):

    norm = np.cumsum(abs_array[:, -1::-1, :], axis=1)[:, -1::-1, :]

    return(abs_array/(norm+reg_param))


def get_depth_average(array_tuple, Nmax=128, Nmin=20):

    rolled = image_roll_3d(array_tuple, Nmax, n=15, d=8, D=16)
    #print(rolled[0][1].shape)

    return(np.mean(rolled[0][1][:, Nmin:, :], axis=1), np.mean(rolled[0][-1][:, Nmin:, :], axis=1))


def get_depth_average_filename(filename, Nmax=128, Nmin=20, reg_param=100.):

    co, abs_array, cross_abs_array = get_array_tuple(filename)

    att_co = get_depth_resolved_coefficients(abs_array, reg_param)
    att_cross = get_depth_resolved_coefficients(cross_abs_array, reg_param)

    co_att, cross_att = get_depth_average((co, att_co, att_cross), Nmax, Nmin)

    return(co_att, cross_att)
