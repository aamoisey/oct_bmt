import numpy as np
from scipy import signal
from functools import partial
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool


def get_vessels_from_filepath(
    filepath,
    shape=(1024, 256, 256),
    dtype='complex64'
):

    oct_data = load_oct_data(filepath,
                             shape=shape,
                             dtype=dtype)

    oct_data_ = oct_data[::2, :, :]

    oct_data_ = correct_phase_parallel(oct_data_,
                                       N=80)

    vessels = get_vesssels_array_parallel(oct_data_)

    return (oct_data, vessels)


def load_oct_data(
    filepath,
    shape=(1024, 256, 256),
    dtype='complex64'
):

    with open(filepath, mode='rb') as file:
        values = file.read()

    values = np.frombuffer(values, dtype=dtype)
    oct_data = np.reshape(values, shape)

    return (oct_data)


def correct_phase(
    oct_data,
    N=80
):

    x = np.arange(-oct_data.shape[1]/2, oct_data.shape[1]/2)
    cdf = np.exp(-1.*x**2/2./N)
    cdf = np.fft.fftshift(cdf)
    cdf = cdf[:, np.newaxis]

    result = np.zeros(oct_data.shape)+1j*np.zeros(oct_data.shape)
    result[0, :, :] = oct_data[0, :, :]

    for i in range(1, oct_data.shape[0]):

        corr = oct_data[i, :, :]*np.conj(result[i-1, :, :])

        corr = np.fft.ifft(np.fft.fft(corr, axis=0)*cdf, axis=0)

        result[i, :, :] = oct_data[i, :, :]*np.exp(-1j*np.angle(corr))

    return (result)


def correct_phase_parallel(
    oct_data,
    N=80
):

    N_stripes = mp.cpu_count()

    correct_phase_ = partial(correct_phase, N=N)

    oct_data_stripes = [oct_data[:, :, i::N_stripes] for i in range(N_stripes)]

    with Pool(N_stripes) as pool:
        oct_data_stripes = pool.map(correct_phase_, [stripe for stripe in oct_data_stripes])

    oct_data_corr = np.zeros(oct_data.shape)+1j*np.zeros(oct_data.shape)

    for i in range(N_stripes):
        oct_data_corr[:, :, i::N_stripes] = oct_data_stripes[i]

    return (oct_data_corr)


def get_vesssels_array_parallel(oct_data_corrected):

    N_stripes = mp.cpu_count()

    oct_data_stripes = [oct_data_corrected[:, :, i::N_stripes] for i in range(N_stripes)]

    with Pool(N_stripes) as pool:
        vessels_stripes = pool.map(get_vesssels, [stripe for stripe in oct_data_stripes])

    vessels = np.zeros(oct_data_corrected.shape)

    for i in range(N_stripes):
        vessels[:, :, i::N_stripes] = np.abs(vessels_stripes[i])

    return (vessels)


def get_vesssels(
    oct_data,
    N=7,
    threshold=2*424,
    # n=3,
    pow=.5
):

    low_pass = signal.fftconvolve(np.abs(oct_data), np.ones((N, 1, 1)), 'same')/(N+.0)

    low_pass = (threshold**2+low_pass**2)**pow

    hp_kernel = gaussian_process(kernel_half_length=3, sgm_f=10.5, kernel_sigma=3, sgm_n=.0)

    result = signal.fftconvolve(oct_data/(np.abs(low_pass)+.1)**1.,
                                hp_kernel[:, np.newaxis, np.newaxis], 'same')

    return (result)


def gaussian_process(
    kernel_half_length=4,
    sgm_f=1.5,
    kernel_sigma=3.,
    sgm_n=3.
):

    N = 2*kernel_half_length+1

    ind = 2

    x = np.arange(N)

    fft_mat = x[:, np.newaxis]-x[np.newaxis, :]

    fft_mat = sgm_f**2*np.exp(-fft_mat**2/2./kernel_sigma**2)

    fft_mat += sgm_n**2*np.eye(N)

    K_star = fft_mat[:, ind]
    K_dbl_star = K_star[ind]
    K_star = np.delete(K_star, ind)

    fft_mat = np.delete(fft_mat, ind, axis=0)
    fft_mat = np.delete(fft_mat, ind, axis=1)
    fft_mat = np.linalg.pinv(fft_mat)

    coeff = np.dot(K_star[np.newaxis, :], fft_mat)

    var = K_dbl_star-np.dot(coeff, K_star[:, np.newaxis])

    coeff = np.squeeze(coeff)

    coeff_ = -1*np.ones(N)
    coeff_[:ind] = coeff[:ind]
    coeff_[ind+1:] = coeff[ind:]

    return (coeff_)


def get_depth_resolved_attenuation(
    abs_oct_array,
    Nwind=32,
    return_snr=False,
    mm_in_pix=7*10**-3,
    noise_region=[20, 10, 40],
    remaining_signal=0,
):

    Nmax = abs_oct_array.shape[1]+1

    intens_oct_array = np.abs(abs_oct_array)**2

    z = np.arange(intens_oct_array.shape[1])
    z = z[np.newaxis, :, np.newaxis]

    #noise = np.mean(array[20:-20, -30:-20, 40:-40])
    noise = np.median(intens_oct_array[noise_region[0]:-noise_region[0], -
                                       noise_region[1]:, noise_region[2]:-noise_region[2]])

    #noise = np.mean(intens_oct_array[20:-20, 5, 40:-40])

    print(noise)
    norm = np.cumsum(intens_oct_array[:, -1::-1, :], axis=1)[:, -1::-1, :]+10**-19

    mu_est = intens_oct_array/(norm+remaining_signal)

    print(np.max(norm))

    H = 1.-noise*(Nmax-z)/norm  # noise*(Nmax-z)*(Nmax+z+1)/2./norm

    SNR = signal.fftconvolve((np.abs(intens_oct_array-noise)**1)/noise**1,
                             np.ones((1, Nwind, Nwind)), 'same')/Nwind/Nwind

    # SNR = signal.fftconvolve((np.abs(intens_oct_array-0*noise)**2)/noise**2,
    #                         np.ones((Nwind, 1, Nwind)), 'same')/Nwind/Nwind

    #SNR[SNR < 0] = 0

    attenuation = mu_est*H*SNR/(H*H*SNR+1)

    attenuation = attenuation/mm_in_pix

    if return_snr:
        return (attenuation, SNR)  # , mu_est/mm_in_pix, H)  # , SNR, mu_est)
    else:
        return (attenuation)


def estimate_mu(
    oct_signal,
    roll_off,
    Nwind=8,
    pix_length=7
):

    N0, N1 = oct_signal.shape  # oct_signal.shape[0]
    z = np.arange(N0)

    oct_signal = np.abs(oct_signal)**2

    #noise = np.mean(oct_signal[-40:-20, 40:-40])
    noise = np.median(oct_signal[20:-20, 5, 40:-40])

    SNR = signal.fftconvolve((np.abs(intens_oct_array-0*noise)**2)/noise**2,
                             np.ones((Nwind, 1, Nwind)), 'same')/Nwind/Nwind

    H = roll_off[np.newaxis, :, np.newaxis]

    oct_signal_wiener = (oct_signal-0*noise)*H*SNR/(H*H*SNR+1)
    oct_signal_naive = (oct_signal-noise)/(H+10**-19)

    norm = np.cumsum(oct_signal[-1::-1, :], axis=0)[-1::-1, :]
    norm_wiener = np.cumsum(oct_signal_wiener[-1::-1, :], axis=0)[-1::-1, :]
    norm_naive = np.cumsum(oct_signal_wiener[-1::-1, :], axis=0)[-1::-1, :]

    vermeer = oct_signal/(norm+10**-29)/2/pix_length
    vermeer_wiener = oct_signal_wiener/(norm_wiener+10**-29)/2/pix_length
    vermeer_naive = oct_signal_naive/(norm_naive+10**-29)/2/pix_length

    mat = np.zeros((N0, N0))
    for i in range(N0):
        mat[i, i:] = roll_off[i:]**1

    aux = np.linalg.lstsq(mat, norm, .0001)  # -noise*(N-z[:, np.newaxis])

    noise_ = np.mean(aux[0][-40:-30, 40:-40])
    SNR_ = signal.fftconvolve((np.abs(aux[0]-1*noise_)**1)/noise_**1,
                              np.ones((Nwind, Nwind)), 'same')/Nwind/Nwind

    oct_signal_estimate = aux[0]/(1+1/SNR_)

    norm_est = np.cumsum((oct_signal_estimate)[-1::-1, :], axis=0)[-1::-1, :]

    att_sign_est = oct_signal_estimate/norm_est

    Hnoise = 1.-noise*(N0-z[:, np.newaxis])/norm
    att_dr0 = vermeer*Hnoise*SNR/(Hnoise*Hnoise*SNR+1)

    '''
    noise_ = noise/norm
    SNR_ = signal.fftconvolve(np.abs(att_sign_est)/noise_,
                              np.ones((Nwind, Nwind)), 'same')/Nwind/Nwind
    '''

    diff_mat = np.diag(np.ones(N0))-1*np.diag(np.ones(N0-1), 1)
    mat = np.diag(roll_off)
    mat = np.dot(mat, diff_mat)

    # norm_est = np.linalg.lstsq(mat, oct_signal-1*noise, .00000001)[0]
    # norm_est = norm_est-np.mean(norm_est[90:100, :], axis=0)[np.newaxis, :]

    K = 120  # 120
    aux = np.diag(np.ones(N0-K))
    aux = np.append(np.zeros((N0-K, K)), aux, axis=1)
    mat = np.append(mat, aux, axis=0)

    right_hand = np.append(oct_signal-1*noise, np.zeros((N0-K, N1)), axis=0)

    norm_est = np.linalg.lstsq(mat, right_hand, .00000001)[0]  # +.005

    Hrloff = roll_off[:, np.newaxis]*norm_est/(norm+10**-19)
    att_dr1 = vermeer*Hrloff*SNR/(Hrloff*Hrloff*SNR+1)

    Hrloff_naive = roll_off[:, np.newaxis]*norm_naive/(norm+10**-19)
    att_dr2 = vermeer*Hrloff_naive*SNR/(Hrloff_naive*Hrloff_naive*SNR+1)

    return (vermeer,
            vermeer_naive,
            vermeer_wiener,
            att_sign_est,
            att_dr0,
            att_dr1,
            att_dr2,
            oct_signal_wiener,
            oct_signal_estimate,
            SNR,
            SNR_,
            norm,
            norm_est)
