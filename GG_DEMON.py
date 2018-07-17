# -*- coding: utf-8 -*-
##############################################################################
# Copyright (c) 2017, Hari Vishnu
#
# This file is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
# 
# Please cite the original paper if you use the algorithm. The algorithm was developed based on the paper 
# V. N. Hari and M. Chitre, “Robust estimation of modulation frequency in impulsive acoustic data,” 
# IEEE Transactions on Aerospace and Electronic Systems, Mar 2017, DOI: 10.1109/TAES.2017.2677621
# HTTP: http://ieeexplore.ieee.org/document/7869289/
# PDF: https://arl.nus.edu.sg/twiki6/pub/ARL/HariVishnu/Robust_estimation_of_modulation_frequency_in_impulsive_acoustic_data.pdf
#
# [fs, data] = wavfile.read(filename)
# [DEMON_PSD, fs_dec, freqrange] = GG_DEMON(data, fs, bp_low = 100.0, bp_high = 4000.0, decfac = 20, nfft = 4096, p = 1.0, p2 = 2.0, filter_flag = True)
#
# Parameters
# - data: raw acoustic data whose modulation frequency you want to detect
# - fs: sampling frequency of raw acoustic data
# - bp_low: lower cut-off of bandpass filter applied on the data (if filter_flag is True)
# - bp_high: upper cut-off of bandpass filter applied on the data (if filter_flag is True)
# - decfac: Decimation factor applied on data, to view PSD in lower sampling frequency. Must be >=1
# - nfft: Number of FFT points in PSD, default is 4096
# - p: Exponential factor of GG or MGG DEMON. p = 2 is conventional DEMON, p<2 is robust, and p must be >0.
# - p2: Exponential factor of template : p2 = 1 applies MGG-DEMON, while p2 = 2 applies GG-DEMON.
# - filter_flag: Whether you want to apply bandpass filtering to the data or not
# 
# Returns: 
# - DEMON_PSD: the magnitude DEMON PSD
# - fs_dec: Sampling frequency of the decimated sequence (max frequency of DEMON PSD)
# - freqrange: Frequency range of the DEMON PSD
##############################################################################

import numpy as np
import matplotlib.pyplot as plt #If you want to plot the spectrum in the function
import scipy.signal as sig

def GG_DEMON(data, fs, bp_low = 100.0, bp_high = 4000.0, decfac = 20.0, nfft = 4096.0, p = 1.0, p2 = 2.0, filter_flag = True):
    
    fs = float(fs)
    data = data.astype(dtype=np.float64);
 
    def butter_bandpass_filter(data, bp_low, bp_high, fs, order=5):
        nyq = 0.5 * fs
        low = bp_low / nyq
        high = bp_high / nyq
        b, a = sig.butter(order, [low, high], btype='band')
        y = sig.lfilter(b, a, data)
        return y
    
    if filter_flag == True:
        if bp_high>fs/2:
            raise ValueError('Enter valid upper limit for bandpass filtering');
        else:
            data_filtered = butter_bandpass_filter(data, bp_low, bp_high, fs, 5)
    elif filter_flag == False:
        data_filtered = data
    else:
        raise ValueError('Wrong input for filter_flag.. must be True or False');
    
    if p<=0:
        raise ValueError('Value of exponential factor p must be >0');
    elif p>2:
        raise Warning('Value of exponential factor p greater than 2 may not be robust');
        
    if p2<=0:
        raise ValueError('Value of exponential factor p2 must be >0');
    elif p2>2:
        raise Warning('Value of exponential factor p2 greater than 2 may not be robust');
        
    if decfac<1:
        raise ValueError('Decimation factor must be >=1');
    
    yRec1 = (np.abs(data_filtered))**p2
    yRec2 = (np.abs(data_filtered))**p
    fs_dec = fs/decfac
    yRec1 = sig.resample(yRec1,int(len(yRec1)/decfac))
    yRec2 = sig.resample(yRec2,int(len(yRec2)/decfac))
    datalength = len(yRec1)
    
    num_windows = int(np.ceil(float(datalength)/float(nfft)))
    yRec1 = np.append(yRec1,np.zeros(nfft*num_windows-datalength))
    yRec2 = np.append(yRec2,np.zeros(nfft*num_windows-datalength))
    
    freqrange = np.linspace(0.0,int(fs_dec/2),int(nfft/2+1))

    a1 = np.fft.fft(np.reshape(yRec1,(nfft,num_windows)),axis=0)
    a1 = (np.abs(a1)**(2.0/p2))*np.exp(1j*np.angle(a1)) 
    #This part is to try to ensure the scaling is not affected due
    #to the nonlinear operation, namely, exponation to power p2
    
    a2 = np.conjugate(np.fft.fft(np.reshape(yRec2,(nfft,num_windows)),axis=0))
    a2 = (np.abs(a2)**(2.0/p))*np.exp(1j*np.angle(a2))
    #This part is to try to ensure the scaling is not affected due
    #to the nonlinear operation, namely, exponation to power p
    
    DEMON_PSD = np.abs(np.mean(np.real(a1*a2),1)/nfft)
    
    ##If you want to plot the spectrum in the function
    plt.figure
    plt.plot(freqrange,10*np.log10(DEMON_PSD[0:int(nfft/2+1)]))
    plt.ylabel('DEMON PSD (dB)')
    plt.xlabel('Modulation frequency (Hz)')
    plt.show()
    
    return DEMON_PSD, fs_dec, freqrange


from scipy.io import wavfile
filename = 'DEMON_Sample.wav'
[fs, data] = wavfile.read(filename)
GG_DEMON(data, fs, 100.0, 4000.0, 40, 1024, 2, 1.0, filter_flag = True)
