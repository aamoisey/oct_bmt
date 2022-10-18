#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:20:59 2017

@author: alexander
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:07:58 2017

@author: Moiseev
"""


#from skimage import morphology


#from sklearn.cluster import MiniBatchKMeans
#from sklearn.decomposition import PCA


#from matplotlib.collections import EllipseCollection

#from OCT import borders
#from OCT import OCT_io as oio




import matplotlib.pyplot as plt
import re
#import xlwt
import os
from sklearn import neighbors
from skimage.restoration import inpaint, denoise_tv_chambolle as tv
import numpy as np
from scipy import signal
from scipy.ndimage import filters as sfilters
from scipy import misc
from skimage import filters, morphology, measure
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

    knn = neighbors.KNeighborsRegressor(N, weights='distance')
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
#    cross = np.abs(co[1::2,:,:])
    co = np.abs(co[::2, :, :])

    aux = signal.fftconvolve(co, np.ones((7, 1, 1)), 'same')
    co_ = co/(aux**2+5000**2)**.5

#    aux = signal.fftconvolve(cross,np.ones((7,1,1)),'same')
#
#    co = co_
#    cross = cross/(aux**2+5000**2)**.5

    return(co_)  # ,co,cross)


############################################################################################
#############################################################################################

def optical_coeff(filename, N=100, N_=40, N0=30):

    #    N = etalon.shape[0]

    x = np.arange(N-N_)

    array = import_SU_bin(filename, (1024, 256, 256), np.complex64)

    array_ = np.abs(array[1::2, :, :])**1  # -np.abs(array[1::2,:,:])**2
    array = np.abs(array[::2, :, :])**1  # +np.abs(array[1::2,:,:])**2
#    array = array[:,:,20:-20]

#    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)

    co_ = get_array_tuple(filename)

    array = sfilters.uniform_filter(array+.0, 5)
    array_ = sfilters.uniform_filter(array_+.0, 5)
#    array_ = np.abs(array_) #array_*(array_>0)+.1*(array_<=0)

#    array_ = array_/array

#    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)

    rolled, supp = image_roll_3d((co_, array, array_), N)

#    rolled,supp = borders.image_roll_3d((co_,aux,aux_))

    mean_co_2d = np.median(rolled[1][:, N_:, N0:-N0], axis=2)
    mean_cross_2d = np.median(rolled[2][:, N_:, N0:-N0], axis=2)

    mean_co_2d_ = np.median(rolled[1][:, N_:, N0:-N0], axis=0)
    mean_cross_2d_ = np.median(rolled[2][:, N_:, N0:-N0], axis=0)

#    aux = sfilters.median_filter(rolled[1],footprint=np.ones((11,11,1)))
#    aux_ = sfilters.median_filter(rolled[2],footprint=np.ones((11,11,1)))

    y = r_hand_(rolled[1][:, N_:, :], rolled[2][:, N_:, :])

    print(3, y[0].shape, y[1].shape)
    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
    aux = np.exp((coeff[-1]-coeff_[-1]))
    mux_mu = (aux-1)/(aux+1)
    C = -(coeff[0]-coeff_[0])/1.
    nu = -coeff[0]/1.

#    ind = (mux_mu>0.)&(mux_mu<1.)&(C>0)&(nu>0)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    coeff_3d = np.append(mux_mu[np.newaxis, :], nu[np.newaxis, :], axis=0)
    coeff_3d = np.append(coeff_3d, C[np.newaxis, :], axis=0)

    y = r_hand_(mean_co_2d, mean_cross_2d)

#    print(2,y[0].shape,y[1].shape)
    coeff = np.polyfit(x, np.log(y[0].T), 1)
    coeff_ = np.polyfit(x, np.log(y[1].T), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
    aux = np.exp((coeff[-1]-coeff_[-1]))
    mux_mu = (aux-1)/(aux+1)
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    ind = (mux_mu > 0.) & (mux_mu < 1.) & (C > 0) & (nu > 0)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    coeff_2d = np.append(mux_mu[np.newaxis, :], nu[np.newaxis, :], axis=0)
    coeff_2d = np.append(coeff_2d, C[np.newaxis, :], axis=0)
    coeff_2d = coeff_2d[:, ind]

    y = r_hand_(mean_co_2d_, mean_cross_2d_)

#    print(1,y[0].shape,y[1].shape)
    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
    aux = np.exp((coeff[-1]-coeff_[-1]))
    mux_mu = (aux-1)/(aux+1)
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    ind = (mux_mu > 0.) & (mux_mu < 1.) & (C > 0) & (nu > 0)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    coeff_2d_ = np.append(mux_mu[np.newaxis, :], nu[np.newaxis, :], axis=0)
    coeff_2d_ = np.append(coeff_2d_, C[np.newaxis, :], axis=0)
    coeff_2d_ = coeff_2d_[:, ind]

    return(coeff_3d, coeff_2d, coeff_2d_)  # ,coeff_1d) #rolled,

# def optical_coeff(filename,N=100,N_=40,N0=30):
#
##    N = etalon.shape[0]
#
#    x = np.arange(N-N_)
#
#    array = import_SU_bin(filename,(1024,256,256),np.complex64)
#
#    array_ = np.abs(array[::2,:,:])**2-np.abs(array[1::2,:,:])**2
#    array = np.abs(array[::2,:,:])**2+np.abs(array[1::2,:,:])**2
##    array = array[:,:,20:-20]
#
##    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
##    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)
#
#    co_ = get_array_tuple(filename)
#
#    array = sfilters.uniform_filter(array+.0,5)
#    array_ = sfilters.uniform_filter(array_+.0,5)
#    array_ = array_*(array_>0)+.1*(array_<=0)
#
##    array_ = array_/array
#
##    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
##    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)
#
#    rolled,supp = image_roll_3d((co_,array,array_),N)
#
##    rolled,supp = borders.image_roll_3d((co_,aux,aux_))
#
#    mean_co_2d = np.median(rolled[1][:,N_:,N0:-N0],axis=2)
#    mean_cross_2d = np.median(rolled[2][:,N_:,N0:-N0],axis=2)
#
#    mean_co_2d_ = np.median(rolled[1][:,N_:,N0:-N0],axis=0)
#    mean_cross_2d_ = np.median(rolled[2][:,N_:,N0:-N0],axis=0)
#
##    aux = sfilters.median_filter(rolled[1],footprint=np.ones((11,11,1)))
##    aux_ = sfilters.median_filter(rolled[2],footprint=np.ones((11,11,1)))
#
#    y = r_hand_(rolled[1][:,N_:,:],rolled[2][:,N_:,:])
#
#    print(3,y[0].shape,y[1].shape)
#    coeff = np.polyfit(x,np.log(y[0]),1)
#    coeff_ = np.polyfit(x,np.log(y[1]),1)
#    print(coeff.shape,coeff_.shape)
#
# mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
#    aux = np.exp((coeff[-1]-coeff_[-1]))
#    mux_mu = (aux-1)/(aux+1)
#    C = (coeff[0]-coeff_[0])/2.
#    nu = -coeff[0]
#
##    ind = (mux_mu>0.)&(mux_mu<1.)&(C>0)&(nu>0)
#
##    mux_mu = 1-np.exp(coeff_[-1]/2.)
##    C = -coeff_[0]/2.
##    nu = -coeff[0]
#
#    coeff_3d = np.append(mux_mu[np.newaxis,:],nu[np.newaxis,:],axis=0)
#    coeff_3d = np.append(coeff_3d,C[np.newaxis,:],axis=0)
#
#    y = r_hand_(mean_co_2d,mean_cross_2d)
#
# print(2,y[0].shape,y[1].shape)
#    coeff = np.polyfit(x,np.log(y[0].T),1)
#    coeff_ = np.polyfit(x,np.log(y[1].T),1)
#    print(coeff.shape,coeff_.shape)
#
# mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
#    aux = np.exp((coeff[-1]-coeff_[-1]))
#    mux_mu = (aux-1)/(aux+1)
#    C = (coeff[0]-coeff_[0])/2.
#    nu = -coeff[0]
#
#    ind = (mux_mu>0.)&(mux_mu<1.)&(C>0)&(nu>0)
#
##    mux_mu = 1-np.exp(coeff_[-1]/2.)
##    C = -coeff_[0]/2.
##    nu = -coeff[0]
#
#    coeff_2d = np.append(mux_mu[np.newaxis,:],nu[np.newaxis,:],axis=0)
#    coeff_2d = np.append(coeff_2d,C[np.newaxis,:],axis=0)
#    coeff_2d = coeff_2d[:,ind]
#
#    y = r_hand_(mean_co_2d_,mean_cross_2d_)
#
# print(1,y[0].shape,y[1].shape)
#    coeff = np.polyfit(x,np.log(y[0]),1)
#    coeff_ = np.polyfit(x,np.log(y[1]),1)
#    print(coeff.shape,coeff_.shape)
#
# mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
#    aux = np.exp((coeff[-1]-coeff_[-1]))
#    mux_mu = (aux-1)/(aux+1)
#    C = (coeff[0]-coeff_[0])/2.
#    nu = -coeff[0]
#
#    ind = (mux_mu>0.)&(mux_mu<1.)&(C>0)&(nu>0)
#
##    mux_mu = 1-np.exp(coeff_[-1]/2.)
##    C = -coeff_[0]/2.
##    nu = -coeff[0]
#
#    coeff_2d_ = np.append(mux_mu[np.newaxis,:],nu[np.newaxis,:],axis=0)
#    coeff_2d_ = np.append(coeff_2d_,C[np.newaxis,:],axis=0)
#    coeff_2d_ = coeff_2d_[:,ind]
#
#    return(coeff_3d,coeff_2d,coeff_2d_) #,coeff_1d) #rolled,


def optical_coeff_vessels(filename, Nmax, Nmin, Nindent):

    #    N = etalon.shape[0]

    x = np.arange(Nmax-Nmin)

    array = import_SU_bin(filename, (1024, 256, 256), np.complex64)

    mip0 = vessels_mip(filename)

    mip_ = tv(mip0, 20)
    mip = tv(mip0, 10)

    bin_vess = mip-mip_ > .0
    bin_vess = 1.-bin_vess
    bin_vess = bin_vess[:, np.newaxis, :]  # N0:-N0]

#    print(bin_vess.shape)

    array_ = np.abs(array[::2, :, :])**2-np.abs(array[1::2, :, :])**2
    array = np.abs(array[::2, :, :])**2+np.abs(array[1::2, :, :])**2
#    array = array[:,:,20:-20]]

#    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)

    co_ = get_array_tuple(filename)

    array = sfilters.uniform_filter(array, 5)
    array_ = sfilters.uniform_filter(array_, 5)
    array_ = array_*(array_ > 0)+.1

#    array_ = array_/array

    rolled, supp = image_roll_3d((co_, array, array_), Nmax)

#    print(bin_vess.shape,rolled[1].shape)

#    rolled,supp = borders.image_roll_3d((co_,aux,aux_))

    aux = np.sum(bin_vess[:, :, Nindent:-Nindent], axis=2)
    mean_co_2d = np.sum((bin_vess*rolled[1])[:, Nmin:, Nindent:-Nindent], axis=2)/aux
    mean_cross_2d = np.sum((bin_vess*rolled[2])[:, Nmin:, Nindent:-Nindent], axis=2)/aux

    aux = np.sum(bin_vess[:, :, Nindent:-Nindent], axis=0)
    mean_co_2d_ = np.sum((bin_vess*rolled[1])[:, Nmin:, Nindent:-Nindent], axis=0)/aux
    mean_cross_2d_ = np.sum((bin_vess*rolled[2])[:, Nmin:, Nindent:-Nindent], axis=0)/aux

#    return(mean_co_2d,mean_cross_2d,mean_co_2d_,mean_cross_2d_,mip0,mip,mip_)

#    aux = sfilters.median_filter(rolled[1],footprint=np.ones((11,11,1)))
#    aux_ = sfilters.median_filter(rolled[2],footprint=np.ones((11,11,1)))

    y = r_hand_(rolled[1][:, Nmin:, :], rolled[2][:, Nmin:, :])

    print(3, y[0].shape, y[1].shape)
    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    mux_mu = (coeff[-1]-coeff_[-1])/2.  # 1-np.exp(coeff_[-1]/2.)
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    coeff_3d = np.append(mux_mu[np.newaxis, :], nu[np.newaxis, :], axis=0)
    coeff_3d = np.append(coeff_3d, C[np.newaxis, :], axis=0)

    y = r_hand_(mean_co_2d, mean_cross_2d)

#    print(2,y[0].shape,y[1].shape)
    coeff = np.polyfit(x, np.log(y[0].T), 1)
    coeff_ = np.polyfit(x, np.log(y[1].T), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    mux_mu = (coeff[-1]-coeff_[-1])/2.  # 1-np.exp(coeff_[-1]/2.)
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    coeff_2d = np.append(mux_mu[np.newaxis, :], nu[np.newaxis, :], axis=0)
    coeff_2d = np.append(coeff_2d, C[np.newaxis, :], axis=0)

    y = r_hand_(mean_co_2d_, mean_cross_2d_)

#    print(1,y[0].shape,y[1].shape)
    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    mux_mu = (coeff[-1]-coeff_[-1])/2.  # 1-np.exp(coeff_[-1]/2.)
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    coeff_2d_ = np.append(mux_mu[np.newaxis, :], nu[np.newaxis, :], axis=0)
    coeff_2d_ = np.append(coeff_2d_, C[np.newaxis, :], axis=0)

    return(coeff_3d, coeff_2d, coeff_2d_)  # ,coeff_1d)


def r_hand_(array_co, array_cross):

    if len(array_co.shape) == 1:
        result = (np.array(array_co), np.array(array_cross))

    elif len(array_co.shape) == 2:
        result = (np.array(array_co), np.array(array_cross))

    else:
        aux = np.ones((array_co.shape[0], array_co.shape[2]))
        result = (np.array((array_co.transpose((0, 2, 1))[aux > 0]).T), np.array(
            (array_cross.transpose((0, 2, 1))[aux > 0]).T))

    return(result)


def coeff_b_scans(filename, Nmax, Nmin, Nindent):

    b_scan = misc.imread(filename, flatten=True)
    b_scan = np.roll(b_scan, -32, axis=1)+.0

    N = np.int(b_scan.shape[0]/2)

    co = b_scan[:N, :]
    cross = b_scan[N:, :]

    mask, _ = support(co, n=15)

    co = 10**(co/50.)
    cross = 10**(cross/50)

#    mask,_ = support(co,n=15)

    x = np.linspace(1, 0, co.shape[0])

#    print(x.shape,mask.shape)

    argmax = np.argmax(x[:, np.newaxis]*mask, axis=0)

    array_ = np.abs(co)**2-np.abs(cross)**2
    array = np.abs(co)**2+np.abs(cross)**2
    array = array[:, Nindent:-Nindent]
    array_ = array_[:, Nindent:-Nindent]
    argmax = argmax[Nindent:-Nindent]

#    array = knn_filter(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter(array_,D=5,threshold=.25,N=3)

    array = sfilters.uniform_filter(array, 5)
    array_ = sfilters.uniform_filter(array_, 5)
    array_ = array_*(array_ > 0)+.1

#    array_ = array_/array

    array_rolled = np.zeros(array.shape)
    array_rolled_ = np.zeros(array_.shape)

#    print(array.shape,array_.shape,argmax.shape)

    for i in range(array.shape[1]):

        #        print(i,argmax[i])
        array_rolled[:, i] = np.roll(array[:, i], -argmax[i])
        array_rolled_[:, i] = np.roll(array_[:, i], -argmax[i])

    y = r_hand_(array_rolled[Nmin:Nmax, :], array_rolled_[Nmin:Nmax, :])
#    y_ = r_hand_(array_rolled_,array_rolled_)

    x = np.arange(Nmax-Nmin)

#    print(x.shape,y[0].shape)

    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    mux_mu = (coeff[-1]-coeff_[-1])/2.  # 1-np.exp(coeff_[-1]/2.)
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    coeff_2d = np.append(mux_mu[np.newaxis, :], nu[np.newaxis, :], axis=0)
    coeff_2d = np.append(coeff_2d, C[np.newaxis, :], axis=0)

    return(coeff_2d)  # co,cross,array,array_,mask,coeff_2d)


def do_in_folder(folder_name, Nmax, Nmin, Nindent, mode='3d'):  # create a template for this

    files_list = os.listdir(folder_name)
    save_npy = folder_name+'/_npy'
    save_xls = folder_name+'/_xls'

    header = np.array(['backscattering relation', 'attenuation', 'forward crosscattering'])
    header = header[np.newaxis, :]

#    slope = np.zeros(30)

    for f in files_list:

        print(f)
        filename = ''.join([folder_name, '/', f])
        split_ext = os.path.splitext(f)[0]
        if os.path.isdir(filename) == 0:

            try:

                if mode == '3d':
                    coeffs = optical_coeff(filename, Nmax, Nmin, Nindent)
                elif mode == '3d_vessels':
                    coeffs = optical_coeff_vessels(filename, Nmax, Nmin, Nindent)
                else:
                    coeffs = coeff_b_scans(filename, Nmax, Nmin, Nindent)

#                misc.imsave(split_ext+'_terrain.bmp',terr)

                if os.path.isdir(save_npy) == 0:
                    os.makedirs(save_npy)
                if os.path.isdir(save_xls) == 0:
                    os.makedirs(save_xls)

                if mode != '2d':
                    np.save(save_npy+'/'+split_ext+'_3d.npy', coeffs[0])
                    np.save(save_npy+'/'+split_ext+'_2d_axis2.npy', coeffs[1])
                    np.save(save_npy+'/'+split_ext+'_2d_axis0.npy', coeffs[2])
    #                np.save(save_npy+'/'+split_ext+'_1d.npy',coeffs[3])

#                    print('header shape',header.shape,coeffs[0].shape)

                    data = np.append(header, coeffs[0][:, :-2:2].T, axis=0)  # [:,::2][:,:-2]
#                    print('append',data.shape)
                    write_xls(save_xls+'/'+split_ext+'_3d.xls', data)

                    data = np.append(header, coeffs[1].T, axis=0)
                    write_xls(save_xls+'/'+split_ext+'_2d_axis2.xls', data)

                    data = np.append(header, coeffs[2].T, axis=0)
                    write_xls(save_xls+'/'+split_ext+'_2d_axis0.xls', data)

                else:
                    np.save(save_npy+'/'+split_ext+'_coeff.npy', coeffs)

                    data = np.append(header, coeffs.T, axis=0)
                    write_xls(save_xls+'/'+split_ext+'_coeffs.xls', data)

            except IOError:
                pass
            except ValueError:
                pass
#            except np.linalg.LinAlgError:
#                pass

        else:
            do_in_folder(filename, Nmax, Nmin, Nindent, mode)

    return()


def do_in_directory(directory_name, N, N_, Nindent):

    files_list = os.listdir(directory_name)

#    data = 'none'
#    names = 'none'

    for f in files_list:

        if os.path.isfile(directory_name+'/'+f):
            pass  # data_from_file(directory_name,f,vmin,vmax) #,data,names) #data,names =
        else:
            do_in_folder(directory_name+'/'+f, N, N_, Nindent)  # ,data,names) #data,names =

    return()  # data,names)

#######################################################################
#######################################################################
#######################################################################


def get_etalon_coeff(filename, N=100, N_=40, N0=30):

    #    N = etalon.shape[0]

    x = np.arange(N-N_)

    array = import_SU_bin(filename, (1024, 256, 256), np.complex64)

#    array_ = np.abs(array[::2,:,:])**2-np.abs(array[1::2,:,:])**2
    array = np.abs(array[::2, :, :])**2+np.abs(array[1::2, :, :])**2

#    array = array[:,:,20:-20]

#    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)

    co_ = get_array_tuple(filename)

    array = sfilters.uniform_filter(array+.0, 5)
#    array_ = sfilters.uniform_filter(array_+.0,5)
#    array_ = array_*(array_>0)+.1*(array_<=0)

#    array_ = array_/array

#    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)

    rolled, supp = image_roll_3d((co_, array), N)

#    rolled,supp = borders.image_roll_3d((co_,aux,aux_))

    mean_co = np.median(np.median(rolled[1][:, N_:, N0:-N0], axis=2), axis=0)


#    print(2,y[0].shape,y[1].shape)
    coeff = np.polyfit(x, np.log(mean_co.T), 1)

    return(coeff[-1])  # ,coeff_1d) #rolled,


def optical_coeff_etalon(filename, etalon_coeff, N=100, N_=40, N0=70):

    #    N = etalon.shape[0]

    x = np.arange(N-N_)

    array = import_SU_bin(filename, (1024, 256, 256), np.complex64)

    array_ = np.abs(array[::2, :, :])**2-np.abs(array[1::2, :, :])**2
    array = np.abs(array[::2, :, :])**2+np.abs(array[1::2, :, :])**2

#    array = array[:,:,20:-20]

#    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)

    co_ = get_array_tuple(filename)

    array = sfilters.uniform_filter(array+.0, 5)
    array_ = sfilters.uniform_filter(array_+.0, 5)
    array_ = array_*(array_ > 0)+.1*(array_ <= 0)

#    array_ = array_/array

#    array = knn_filter_3d(array,D=5,threshold=.25,N=3)
#    array_ = knn_filter_3d(array_,D=5,threshold=.25,N=3)

    rolled, supp = image_roll_3d((co_, array, array_), N)

#    rolled,supp = borders.image_roll_3d((co_,aux,aux_))

    mean_co_2d = np.median(rolled[1][:, N_:, N0:-N0], axis=2)
    mean_cross_2d = np.median(rolled[2][:, N_:, N0:-N0], axis=2)

    mean_co_2d_ = np.median(rolled[1][:, N_:, N0:-N0], axis=0)
    mean_cross_2d_ = np.median(rolled[2][:, N_:, N0:-N0], axis=0)

#    aux = sfilters.median_filter(rolled[1],footprint=np.ones((11,11,1)))
#    aux_ = sfilters.median_filter(rolled[2],footprint=np.ones((11,11,1)))

    y = r_hand_(rolled[1][:, N_:, :], rolled[2][:, N_:, :])

    print(3, y[0].shape, y[1].shape)
    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
    mu = np.exp((coeff[-1]-etalon_coeff))  # coeff_[-1]))
    mux = np.exp((coeff_[-1]-etalon_coeff))
    mu = (mu+mux)/2.
    mux = mu-mux
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]


#    ind = (mux_mu>0.)&(mux_mu<1.)&(C>0)&(nu>0)

#    mux_mu = 1-np.exp(coeff_[-1]/2.)
#    C = -coeff_[0]/2.
#    nu = -coeff[0]

    coeff_3d = np.append(mu[np.newaxis, :], mux[np.newaxis, :], axis=0)
    coeff_3d = np.append(coeff_3d, nu[np.newaxis, :], axis=0)
    coeff_3d = np.append(coeff_3d, C[np.newaxis, :], axis=0)

    y = r_hand_(mean_co_2d, mean_cross_2d)

#    print(2,y[0].shape,y[1].shape)
    coeff = np.polyfit(x, np.log(y[0].T), 1)
    coeff_ = np.polyfit(x, np.log(y[1].T), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
    mu = np.exp((coeff[-1]-etalon_coeff))  # coeff_[-1]))
    mux = np.exp((coeff_[-1]-etalon_coeff))
    mu = (mu+mux)/2.
    mux = mu-mux
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    ind = (mu > .0) & (mux > .0) & (C > 0) & (nu > 0)

    coeff_2d = np.append(mu[np.newaxis, :], mux[np.newaxis, :], axis=0)
    coeff_2d = np.append(coeff_2d, nu[np.newaxis, :], axis=0)
    coeff_2d = np.append(coeff_2d, C[np.newaxis, :], axis=0)
    coeff_2d = coeff_2d[:, ind]

    y = r_hand_(mean_co_2d_, mean_cross_2d_)

#    print(1,y[0].shape,y[1].shape)
    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
    mu = np.exp((coeff[-1]-etalon_coeff))  # coeff_[-1]))
    mux = np.exp((coeff_[-1]-etalon_coeff))
    mu = (mu+mux)/2.
    mux = mu-mux
    C = (coeff[0]-coeff_[0])/2.
    nu = -coeff[0]

    ind = (mu > .0) & (mux > .0) & (C > 0) & (nu > 0)

    coeff_2d_ = np.append(mu[np.newaxis, :], mux[np.newaxis, :], axis=0)
    coeff_2d_ = np.append(coeff_2d_, nu[np.newaxis, :], axis=0)
    coeff_2d_ = np.append(coeff_2d_, C[np.newaxis, :], axis=0)
    coeff_2d_ = coeff_2d_[:, ind]

    return(coeff_3d, coeff_2d, coeff_2d_)  # ,coeff_1d) #rolled,


def do_in_folder_etalon(folder_name, filename_etalon, Nmax, Nmin, Nindent, mode='3d'):  # create a template for this

    files_list = os.listdir(folder_name)
    save_npy = folder_name+'/_npy'
    save_xls = folder_name+'/_xls'

    header = np.array(['backscattering relation', 'attenuation', 'forward crosscattering'])
    header = header[np.newaxis, :]

#    slope = np.zeros(30)

    etalon_coeff = get_etalon_coeff(filename_etalon, Nmax, Nmin, Nindent)

    for f in files_list:

        print(f)
        filename = ''.join([folder_name, '/', f])
        split_ext = os.path.splitext(f)[0]
        if os.path.isdir(filename) == 0:

            try:

                if mode == '3d':
                    coeffs = optical_coeff_etalon(filename, etalon_coeff, Nmax, Nmin, Nindent)
                elif mode == '3d_vessels':
                    coeffs = optical_coeff_vessels(filename, Nmax, Nmin, Nindent)
                else:
                    coeffs = coeff_b_scans(filename, Nmax, Nmin, Nindent)

#                misc.imsave(split_ext+'_terrain.bmp',terr)

                if os.path.isdir(save_npy) == 0:
                    os.makedirs(save_npy)
                if os.path.isdir(save_xls) == 0:
                    os.makedirs(save_xls)

                if mode != '2d':
                    np.save(save_npy+'/'+split_ext+'_3d.npy', coeffs[0])
                    np.save(save_npy+'/'+split_ext+'_2d_axis2.npy', coeffs[1])
                    np.save(save_npy+'/'+split_ext+'_2d_axis0.npy', coeffs[2])
    #                np.save(save_npy+'/'+split_ext+'_1d.npy',coeffs[3])

#                    print('header shape',header.shape,coeffs[0].shape)

                    data = np.append(header, coeffs[0][:, :-2:2].T, axis=0)  # [:,::2][:,:-2]
#                    print('append',data.shape)
                    write_xls(save_xls+'/'+split_ext+'_3d.xls', data)

                    data = np.append(header, coeffs[1].T, axis=0)
                    write_xls(save_xls+'/'+split_ext+'_2d_axis2.xls', data)

                    data = np.append(header, coeffs[2].T, axis=0)
                    write_xls(save_xls+'/'+split_ext+'_2d_axis0.xls', data)

                else:
                    np.save(save_npy+'/'+split_ext+'_coeff.npy', coeffs)

                    data = np.append(header, coeffs.T, axis=0)
                    write_xls(save_xls+'/'+split_ext+'_coeffs.xls', data)

            except IOError:
                pass
            except ValueError:
                pass
#            except np.linalg.LinAlgError:
#                pass

        else:
            do_in_folder_etalon(filename, filename_etalon, Nmax, Nmin, Nindent, mode)

    return()


def do_in_directory_etalon(directory_name, filename_etalon, N, N_, Nindent):

    files_list = os.listdir(directory_name)

#    data = 'none'
#    names = 'none'

    for f in files_list:

        if os.path.isfile(directory_name+'/'+f):
            pass  # data_from_file(directory_name,f,vmin,vmax) #,data,names) #data,names =
        else:
            do_in_folder_etalon(directory_name+'/'+f, filename_etalon, N,
                                N_, Nindent)  # ,data,names) #data,names =

    return()  # data,names)

#######################################################################
#######################################################################
#######################################################################


def write_xls(file_path, data):
    pass

    return()

'''
    N, M = data.shape

    book = xlwt.Workbook()
#    print('book')
    sheet1 = book.add_sheet('sheet1')
#    print('sheet')

    for i in range(N):
        #        print('write data',i)
        for j in range(M):
            sheet1.write(i, j, data[i, j])

    book.save(file_path)
'''

def write_csv(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            f.write(",".join(line) + "\n")


def data_from_folder(folder_name, three_D=True):

    files_list = os.listdir(folder_name)

#    N = len(files_list)

#    data = np.load(folder_name+'/'+files_list[0])
#    data = data[:,:,np.newaxis]

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

    return(data)

#######################################################################
#######################################################################
#######################################################################


def real_cumul_angle(array, N=80, amplitude=True):  # split to "real" and "image" is unnecessary. Rewrite

    result = np.zeros(array.shape)+1j*np.zeros(array.shape)

    angle = np.zeros(array.shape)+1j*np.zeros(array.shape)

    result[0, :, :] = array[0, :, :]

    if N > 0:
        x = np.arange(-array.shape[1]/2, array.shape[1]/2)
        cdf = np.exp(-1.*x**2/2./N)
    #    cdf = N*np.sin(1.*x/N)/x
        cdf = np.fft.fftshift(cdf)
    #    cdf[np.isnan(cdf)] = 1
        cdf = cdf[:, np.newaxis]

    for i in range(1, array.shape[0]):

        aux = array[i, :, :]*np.conj(result[i-1, :, :])
        angle[i, :, :] = aux

        real = np.real(aux)
        imag = np.imag(aux)

        if N > 0:

            real = np.fft.ifft(np.fft.fft(real, axis=0)*cdf, axis=0)
            imag = np.fft.ifft(np.fft.fft(imag, axis=0)*cdf, axis=0)
            if amplitude == True:

                aux_amp = sfilters.gaussian_filter(np.abs(result[i-1, :, :]), 2*5+1)
#                aux_amp = aux_amp*(aux_amp>th)+th*(aux_amp<th)
                aux_amp_ = sfilters.gaussian_filter(np.abs(array[i, :, :]), 2*5+1)
#                aux_amp_ = aux_amp_*(aux_amp_>th)+th*(aux_amp_<th)

                aux_amp /= aux_amp_

                print(np.max(aux_amp), np.mean(np.abs(array[i, :, :])))
            aux = np.angle(real+1j*imag)

        else:
            real = np.sum(real[35:125, :], axis=0)
            imag = np.sum(imag[35:125, :], axis=0)
            aux = np.angle(real+1j*imag)
            aux = aux[np.newaxis, :]

        # np.real(array[i,:,:])*np.cos(aux)-1j*np.imag(array[i,:,:])*np.sin(aux) #array[i,:,:]*np.exp(-1j*aux)
        result[i, :, :] = array[i, :, :]*np.exp(-1j*aux)

        if amplitude == True:
            result[i, :, :] *= aux_amp

    return(result)  # ,angle)


def gaussian_process(Nh=4, sgm_f=1.5, l=3., sgm_n=3.):

    N = 2*Nh+1

    ind = 2

    x = np.arange(N)

    fft_mat = x[:, np.newaxis]-x[np.newaxis, :]

    fft_mat = sgm_f**2*np.exp(-fft_mat**2/2./l**2)

    fft_mat += sgm_n**2*np.eye(N)

    K_star = fft_mat[:, ind]
    K_dbl_star = K_star[ind]
    K_star = np.delete(K_star, ind)

    fft_mat = np.delete(fft_mat, ind, axis=0)
    fft_mat = np.delete(fft_mat, ind, axis=1)
    fft_mat = np.linalg.pinv(fft_mat)

    coeff = np.dot(K_star[np.newaxis, :], fft_mat)
    print(coeff.shape)

    var = K_dbl_star-np.dot(coeff, K_star[:, np.newaxis])
    print(var)

    coeff = np.squeeze(coeff)

    coeff_ = -1*np.ones(N)
    coeff_[:ind] = coeff[:ind]
    coeff_[ind+1:] = coeff[ind:]

    return(var, coeff_)  # ,fft_mat)


def high_pass(array, wp=.2, nt=5, wnd='hamming', n=7, shift=True, sgm_n=30.):

    if shift == False:
        b = signal.firwin(nt, wp, pass_zero=False, window=wnd)

    else:

        _, b = gaussian_process(nt, wp, n, sgm_n)

    N = b.shape[0]

    print(b)

    result = np.zeros(array.shape)+1j*np.zeros(array.shape)

    result[:N, :, :] = array[:N, :, :]

#    aux_array =

    for i in range(N, array.shape[0]):

        #        result[i,:,:] = np.sum(b[::-1][:,np.newaxis,np.newaxis]*array[i-N+1:i+1,:,:],axis=0)
        result[i, :, :] = np.sum(b[:, np.newaxis, np.newaxis]*array[i-N+1:i+1, :, :], axis=0)

    return(result)


def high_pass_nonlin(array, N=11, threshold=500, n=3, hp=True, pow_=.5):  # ,7,424,3,False,.5)

    low_pass = signal.fftconvolve(np.abs(array), np.ones((N, 1, 1)), 'same')/(N+.0)

    low_pass = (threshold**2+low_pass**2)**pow_

    if hp == True:
        result = high_pass(array/(np.abs(low_pass)+.1)**1.0, .9,
                           2*n+1, wnd='hamming', n=6, shift=False)
    else:
        result = high_pass(array/(np.abs(low_pass)+.1)**1., 10.5, n, n=2.5, shift=True, sgm_n=00.)

    return(result)


def vessels_mip(filename):  # ,supp):

    #    print('load')
    #    A = oio.flow_quad_patient(filename,size=(4096,256,512))
    array = import_SU_bin(filename, mode='complex64', size=(1024, 256, 256))[::2, :, :]
    A = np.zeros(array.shape)+1j*np.zeros(array.shape)
    print('equal_phase')
    for i in range(16):
        A[:, :, i::16] = real_cumul_angle(array[:, :, i::16], N=80, amplitude=False)
    print('vasc_process')
    for i in range(16):
        A[:, :, i::16] = high_pass_nonlin(A[:, :, i::16], 7, 2*424, 3, False, .5)

#    aux = np.sum(np.sum(supp,axis=-1),axis=0)
#    indxs = np.where(aux>0)
#    N = indxs[0][-1]-indxs[0][0]
#    print(N)

    mip = np.max(np.abs(A)[:, 10:130, :], axis=1)

    return(mip)


def knn_filter(image, D=5, threshold=.5, N=5):

    background = sfilters.uniform_filter(image, D)

    mask = (np.abs(image-background)/background < threshold)
    mask = (image > 0) & mask
    mask = mask.ravel()
    vals = image.ravel()

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)

    vec = np.append(X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis], axis=1)

    knr = neighbors.KNeighborsRegressor(n_neighbors=N, weights='distance', algorithm='auto',
                                        leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)

    knr.fit(vec[mask == 1], vals[mask == 1])

    result = np.zeros(vals.shape)
    result[mask == 1] = vals[mask == 1]
    result[mask == 0] = knr.predict(vec[mask == 0])

    return(result.reshape(image.shape))


def knn_filter_3d(volume, D=5, threshold=.5, N=5):

    result = np.zeros(volume.shape)

    for i in range(volume.shape[0]):

        #        print(i)
        result[i, :, :] = knn_filter(volume[i, :, :], D, threshold, N)

    return(result)


def knn_fill(image, mask=None, val_max=1., N=5):

    if mask == None:
        mask = (image > .0) & (image < val_max)
    mask = mask.ravel()
    vals = image.ravel()

    if np.sum(mask == 0) > 0:

        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        X, Y = np.meshgrid(x, y)

        vec = np.append(X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis], axis=1)

        knr = neighbors.KNeighborsRegressor(n_neighbors=N, weights='distance', algorithm='auto',
                                            leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)

        knr.fit(vec[mask == 1], vals[mask == 1])

        result = np.zeros(vals.shape)
        result[mask == 1] = vals[mask == 1]
        result[mask == 0] = knr.predict(vec[mask == 0])
    else:
        result = image

    return(result.reshape(image.shape))


def border_from_top(filename, N=180):

    array = import_SU_bin(filename, (1024, 256, 256), np.complex64)
#    array = array[:,:,20:-20]

    co_ = np.abs(array[::2, :, :])

    aux = signal.fftconvolve(co_, np.ones((7, 1, 1)), 'same')
    co_ = co_/(aux**2+5000**2)**.5

    rolled, supp = image_roll_3d((co_, (np.abs(array[::2, :, :])), (np.abs(array[1::2, :, :]))), N)

#    co = cube_data(rolled[1])
#    cross = cube_data(rolled[2])

    return(rolled[1], rolled[2])

############################################################################
############################################################################
############################################################################


# ,data='none',names='none'):
def data_from_file(directory, filename, vmin=[-6, .05, -7], vmax=[-1, .15, -1], colormap='jet', etalon=False, pix_to_mm=.007):

    cmap = plt.get_cmap(colormap)

    if (re.search('3d.npy', filename)):  # 'axis0.npy':

        data_ = np.load(directory+'/'+filename)

        print(np.min(data_, axis=1), np.max(data_, axis=1))

#        mask = np.all(data_>.0,axis=0)
#        mask = np.ones(data_[0,:])>0 #data_[0,:]>0
#        if etalon==False:
#            mask = mask&(data_[0,:]<1.)
#
#        if etalon==False:
#
#            aux = data_[0,:].reshape((512,-1))
#            aux = knn_fill(aux,mask)
#            aux = np.log(aux)[::2,:]
#            aux[aux<vmin[0]] = vmin[0]
#            aux[aux>vmax[0]] = vmax[0]
#
#    #        print(np.min(aux),np.max(aux))
#
#            aux = aux-np.min(aux)
#            aux = aux/np.max(aux)
#
#            aux = signal.resample(aux[:,10:-20],256,axis=1)
#
#            misc.imsave(directory+'/'+filename[:-6]+'mu_mux.bmp',cmap(aux))
#
#        else:
#
#            aux = data_[0,:].reshape((512,-1))
#            aux = knn_fill(aux,mask)
#            aux = aux[::2,:]
#            aux[aux<vmin[0]] = vmin[0]
#            aux[aux>vmax[0]] = vmax[0]
#
#    #        print(np.min(aux),np.max(aux))
#
#            aux = aux-np.min(aux)
#            aux = aux/np.max(aux)
#
#            aux = signal.resample(aux[:,10:-20],256,axis=1)
#
#            misc.imsave(directory+'/'+filename[:-6]+'mu.bmp',cmap(aux))
#
#            aux = data_[1,:].reshape((512,-1))
#            aux = knn_fill(aux,mask)
#            aux = np.log(aux)[::2,:]
#            aux[aux<vmin[1]] = vmin[1]
#            aux[aux>vmax[1]] = vmax[1]
#
#    #        print(np.min(aux),np.max(aux))
#
#            aux = aux-np.min(aux)
#            aux = aux/np.max(aux)
#
#            aux = signal.resample(aux[:,10:-20],256,axis=1)
#
#            misc.imsave(directory+'/'+filename[:-6]+'mux.bmp',cmap(aux))

        aux = data_[-2, :].reshape((512, -1))
#        aux = knn_fill(aux,mask)
        aux = (aux[::2, :]+aux[1::2, :])/2./pix_to_mm
        aux[aux < vmin[-2]] = vmin[-2]
        aux[aux > vmax[-2]] = vmax[-2]

#        print(np.min(aux),np.max(aux))

        aux = aux-vmin[-2]  # np.min(aux)
        aux = aux/(vmax[-2]-vmin[-2])  # np.max(aux)

        aux = signal.resample(aux[:, 10:-20], 256, axis=1)

        misc.imsave(directory+'/'+filename[:-6]+'att.bmp', cmap(aux))

        aux = data_[-1, :].reshape((512, -1))
#        aux = knn_fill(aux) #,mask)
        print(np.min(aux), np.max(aux))
        aux = (aux[::2, :]+aux[1::2, :])/2./pix_to_mm  # np.log(aux)[::2,:]
        aux[aux < vmin[-1]] = vmin[-1]
        aux[aux > vmax[-1]] = vmax[-1]

#        print(np.min(aux),np.max(aux))

        aux = aux-vmin[-1]  # np.min(aux)
        aux = aux/(vmax[-1]-vmin[-1])  # np.max(aux)

        aux = signal.resample(aux[:, 10:-20], 256, axis=1)

        misc.imsave(directory+'/'+filename[:-6]+'cross.bmp', cmap(aux))

#        if np.any((data=='none')&(names=='none')):
#            data = data_[:,:,np.newaxis]
#            names = filename
#        else:
#            data = np.append(data,data_[:,:,np.newaxis],axis=-1)
#            names = np.append(names,filename)

    return()  # data,names)


# ,data='none',names='none'):
def data_from_folder_(folder_name, vmin=[-6, .05, -7], vmax=[-1, .15, -1], colormap='jet', etalon=False):

    files_list = os.listdir(folder_name)

#    N = len(files_list)

#    data = np.load(folder_name+'/'+files_list[0])
#    data = data[:,:,np.newaxis]

#    data = 'none'
#    names = 'none'

    for f in files_list:
        #    for n in range(1,N):

        #        f = files_list[n]

        #        print(f[-6:])
        if (os.path.isfile(folder_name+'/'+f)):
            if (re.search('3d.npy', f)):  # 'axis0.npy':

                print(f)

                # ,data,names) ##data,names =
                data_from_file(folder_name, f, vmin, vmax, colormap, etalon)
            else:
                pass

        else:
            pass

    return()  # data,names)


def data_from_directory(directory_name, vmin=[-6, .05, -7], vmax=[-1, .15, -1], colormap='jet', etalon=False):

    files_list = os.listdir(directory_name)

#    data = 'none'
#    names = 'none'

    for f in files_list:

        if os.path.isfile(directory_name+'/'+f):
            data_from_file(directory_name, f, vmin, vmax, colormap,
                           etalon)  # ,data,names) #data,names =
        else:
            data_from_folder_(directory_name+'/'+f, vmin, vmax, colormap,
                              etalon)  # ,data,names) #data,names =

    return()  # data,names)

############################################################################################
############################################################################################
############################################################################################


def optical_coeff_depth_(filename, N=100, N_=40, N0=30, reg_par=50.):  # 1500.):

    array = import_SU_bin(filename, (1024, 256, 256), np.complex64)

    array_ = np.abs(array[1::2, :, :])**1  # -np.abs(array[1::2,:,:])**2
    array = np.abs(array[::2, :, :])**1  # +np.abs(array[1::2,:,:])**2

    co_ = get_array_tuple(filename)

    array = sfilters.median_filter(array+.0, (3, 3, 3))
    array_ = sfilters.median_filter(array_+.0, (3, 3, 3))

    rolled, _ = image_roll_3d((co_, array, array_), 200)

    norm_co = np.cumsum(rolled[1][:, -1::-1, :], axis=1)[:, -1::-1, :]
    norm_cross = np.cumsum(rolled[2][:, -1::-1, :], axis=1)[:, -1::-1, :]

    coeff_co = .63*rolled[1]/(norm_co+reg_par)
    coeff_cross = .63*rolled[2]/(norm_cross+reg_par)

    diff_coeff = coeff_co-coeff_cross

    mean_coeff_co = np.mean(coeff_co[:, N_:N, :], axis=1)
    mean_coeff_cross = np.mean(coeff_cross[:, N_:N, :], axis=1)
    mean_diff_coeff = np.mean(diff_coeff[:, N_:N, :], axis=1)

    return(coeff_co, coeff_cross, diff_coeff, mean_coeff_co, mean_coeff_cross, mean_diff_coeff)


def optical_coeff_depth(filename, N=100, N_=40, N0=30, reg_par=50.):  # 1500.):

    array = import_SU_bin(filename, (1024, 256, 256), np.complex64)

    array_ = np.abs(array[1::2, :, :])**1  # -np.abs(array[1::2,:,:])**2
    array = np.abs(array[::2, :, :])**1  # +np.abs(array[1::2,:,:])**2

    co_ = get_array_tuple(filename)

    array = sfilters.median_filter(array+.0, (3, 3, 3))
    array_ = sfilters.median_filter(array_+.0, (3, 3, 3))

    rolled, _ = image_roll_3d((co_, array, array_), 200)

    x = np.arange(60)

    smoothed1 = sfilters.uniform_filter(rolled[1], 11)
    smoothed2 = sfilters.uniform_filter(rolled[2], 11)
    y = r_hand_(smoothed1[:, 20:80, :], smoothed2[:, 20:80, :])

    print(3, y[0].shape, y[1].shape)
    coeff = np.polyfit(x, np.log(y[0]), 1)
    coeff_ = np.polyfit(x, np.log(y[1]), 1)
    print(coeff.shape, coeff_.shape)

#    mux_mu = (coeff[-1]-coeff_[-1])/2. #1-np.exp(coeff_[-1]/2.)
    C = -coeff_[0]/1.
    nu = -coeff[0]/1.

    print(np.mean(C), np.mean(nu))

    norm_co = np.cumsum(rolled[1][:, -1::-1, :], axis=1)[:, -1::-1, :]
    norm_cross = np.cumsum(rolled[2][:, -1::-1, :], axis=1)[:, -1::-1, :]

    coeff_co = rolled[1]/(norm_co+reg_par)  # .63*
    coeff_cross = rolled[2]/(norm_cross+reg_par)  # .63*

    mean_coeff_co_all = np.mean(coeff_co[:, 20:80, :], axis=1)
    mean_coeff_cross_all = np.mean(coeff_cross[:, 20:80, :], axis=1)

    print(np.mean(mean_coeff_cross_all), np.mean(mean_coeff_co_all))

    prop_co = np.polyfit(mean_coeff_co_all.ravel(), nu.ravel(), 1)
    prop_cross = np.polyfit(mean_coeff_cross_all.ravel(), C.ravel(), 1)

    method_proportionality_coeff0 = (prop_co[0]+prop_cross[0])/2.
    method_proportionality_coeff1 = (prop_co[1]+prop_cross[1])/2.

    print(method_proportionality_coeff0, method_proportionality_coeff1)
    print(np.median(method_proportionality_coeff0*coeff_co[:, 20:80, :]+method_proportionality_coeff1), np.median(
        method_proportionality_coeff0*coeff_cross[:, 20:80, :]+method_proportionality_coeff1))

    coeff_co = prop_co[0]*coeff_co+prop_co[1]
    coeff_cross = prop_cross[0]*coeff_cross+prop_cross[1]

    diff_coeff = coeff_co-coeff_cross

    mean_coeff_co = np.mean(coeff_co[:, N_:N, :], axis=1)
    mean_coeff_cross = np.mean(coeff_cross[:, N_:N, :], axis=1)
    mean_diff_coeff = np.mean(diff_coeff[:, N_:N, :], axis=1)

    # nu,C,mean_coeff_co_all,mean_coeff_cross_all) #

    # return(np.mean(coeff_co[:, 20:80, :], axis=1), np.mean(coeff_cross[:, 20:80, :], axis=1), nu, C)
    return(coeff_co, coeff_cross, diff_coeff, mean_coeff_co, mean_coeff_cross, mean_diff_coeff)


def do_in_folder_depth(folder_name, Nmax, Nmin, Nindent, pix_to_mm=.007):  # create a template for this

    files_list = os.listdir(folder_name)
    save_npy = folder_name+'/_dr_npy'
    save_npy3d = folder_name+'/_dr_npy3d'
    save_xls = folder_name+'/_dr_xls'

    header = np.array(['diff attenuation', 'attenuation co', 'attenuation cross'])
    header = header[np.newaxis, :]

#    slope = np.zeros(30)

    for f in files_list:

        print(f)
        filename = ''.join([folder_name, '/', f])
        split_ext = os.path.splitext(f)[0]
        if os.path.isdir(filename) == 0:

            try:

                coeffs = optical_coeff_depth(filename, Nmax, Nmin, Nindent)

#                misc.imsave(split_ext+'_terrain.bmp',terr)

                if os.path.isdir(save_npy) == 0:
                    os.makedirs(save_npy)
                if os.path.isdir(save_npy3d) == 0:
                    os.makedirs(save_npy3d)
                if os.path.isdir(save_xls) == 0:
                    os.makedirs(save_xls)

                coeff_array = np.append(coeffs[5].ravel()[np.newaxis, :],
                                        coeffs[3].ravel()[np.newaxis, :], axis=0)
                coeff_array = np.append(coeff_array, coeffs[4].ravel()[np.newaxis, :], axis=0)

                np.save(save_npy3d+'/'+split_ext+'_'+np.str(Nmax) +
                        '-'+np.str(Nmin)+'_att_diff.npy', coeffs[0])
                np.save(save_npy3d+'/'+split_ext+'_'+np.str(Nmax) +
                        '-'+np.str(Nmin)+'_att_co.npy', coeffs[1])
                np.save(save_npy3d+'/'+split_ext+'_'+np.str(Nmax) +
                        '-'+np.str(Nmin)+'_att_cross.npy', coeffs[2])
                # np.save(save_npy+'/'+split_ext+'_att_co.npy',coeffs[2])
                # np.save(save_npy+'/'+split_ext+'_att_diff.npy',coeffs[3])

                np.save(save_npy+'/'+split_ext+'_'+np.str(Nmax) +
                        '-'+np.str(Nmin)+'_coeff.npy', coeff_array)

#                np.save(save_npy+'/'+split_ext+'_1d.npy',coeffs[3])

#                    print('header shape',header.shape,coeffs[0].shape)

                #coeff_array = np.append(coeffs[2].ravel()[:,np.newaxis],coeffs[3].ravel()[:,np.newaxis],axis=1)

                data = np.append(header, (coeff_array[:, ::16].T) /
                                 2/pix_to_mm, axis=0)  # [:,::2][:,:-2]
#                    print('append',data.shape)
                write_xls(save_xls+'/'+split_ext+'_'+np.str(Nmax) +
                          '-'+np.str(Nmin)+'_depth_resolved.xls', data)

            except IOError:
                pass
            except ValueError:
                pass
#            except np.linalg.LinAlgError:
#                pass

        else:
            do_in_folder_depth(filename, Nmax, Nmin, Nindent)

    return()


# ,data='none',names='none'):
def data_from_file_depth(directory, filename, vmin=[.003, .003, -.02], vmax=[.03, .03, .02], colormap='jet', etalon=False, pix_to_mm=.007):

    cmap = plt.get_cmap(colormap)

    if (re.search('.npy', filename)):

        data_ = np.load(directory+'/'+filename)

        print(np.min(data_, axis=1), np.max(data_, axis=1))

        aux = data_[0, :].reshape((512, -1))
    #        aux = knn_fill(aux,mask)
        aux = (aux[::2, :]+aux[1::2, :])/2/pix_to_mm  # *.63 #/2./.007
        aux[aux < vmin[0]] = vmin[0]
        aux[aux > vmax[0]] = vmax[0]

    #        print(np.min(aux),np.max(aux))

        aux = aux-vmin[0]  # np.min(aux)
        aux = aux/(vmax[0]-vmin[0])  # np.max(aux)

        aux = signal.resample(aux[:, 10:-20], 256, axis=1)

        #misc.imsave(directory+'/'+filename[:-9]+'att diff.bmp', cmap(aux))
        plt.imsave(directory+'/'+filename[:-9]+'att diff.bmp', cmap(aux))

        aux = data_[1, :].reshape((512, -1))
    #        aux = knn_fill(aux) #,mask)
        print(np.min(aux), np.max(aux))
        aux = (aux[::2, :]+aux[1::2, :])/2/pix_to_mm  # /2./.007 #np.log(aux)[::2,:]
        aux[aux < vmin[1]] = vmin[1]
        aux[aux > vmax[1]] = vmax[1]

    #        print(np.min(aux),np.max(aux))

        aux = aux-vmin[1]  # np.min(aux)
        aux = aux/(vmax[1]-vmin[1])  # np.max(aux)

        aux = signal.resample(aux[:, 10:-20], 256, axis=1)

        plt.imsave(directory+'/'+filename[:-9]+'att co.bmp', cmap(aux))

        aux = data_[2, :].reshape((512, -1))
    #        aux = knn_fill(aux) #,mask)
        print(np.min(aux), np.max(aux))
        aux = (aux[::2, :]+aux[1::2, :])/2/pix_to_mm  # /2./.007 #np.log(aux)[::2,:]
        aux[aux < vmin[2]] = vmin[2]
        aux[aux > vmax[2]] = vmax[2]

    #        print(np.min(aux),np.max(aux))

        aux = aux-vmin[2]  # np.min(aux)
        aux = aux/(vmax[2]-vmin[2])  # np.max(aux)

        aux = signal.resample(aux[:, 10:-20], 256, axis=1)

        plt.imsave(directory+'/'+filename[:-9]+'att cross.bmp', cmap(aux))

    return()  # data,names)


def data_from_directory_depth(directory_name, vmin=[.003, .003, -.02], vmax=[.03, .03, .02], colormap='jet', etalon=False, pix_to_mm=.007):

    files_list = os.listdir(directory_name)

#    data = 'none'
#    names = 'none'

    for f in files_list:

        if os.path.isfile(directory_name+'/'+f):
            data_from_file_depth(directory_name, f, vmin, vmax, colormap,
                                 etalon, pix_to_mm)  # ,data,names) #data,names =
        else:
            data_from_folder_depth(directory_name+'/'+f, vmin, vmax, colormap,
                                   etalon, pix_to_mm)  # ,data,names) #data,names =

    return()  # data,names)


# ,data='none',names='none'):
def data_from_folder_depth(folder_name, vmin=[.003, .003, -.02], vmax=[.03, .03, .02], colormap='jet', etalon=False, pix_to_mm=.007):

    files_list = os.listdir(folder_name)

    for f in files_list:

        data_from_file_depth(folder_name, f, vmin, vmax, colormap,
                             etalon, pix_to_mm)  # ,data,names) ##data,names =

    return()  # data,names)
