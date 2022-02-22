# FOR PLOTTING, SAVING, AND CROPPING THE SPECTRUM TO NIR

# Basic libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt


savedir = 'output/'

def visualize(wv, input, output, sample):
    '''
    Description
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.title.set_text('Input: Spectrum')
    ax1.plot(wv, input[sample], 'b')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transmittance')
    ax1.set_ylim(0,1)

    ax2.title.set_text('Output: Envelope')
    ax2.plot(wv, input[sample], 'b', linestyle ='dashed', alpha=0.25)
    ax2.plot(wv, output[sample], 'r')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Transmittance')
    ax2.set_ylim(0,1)
    plt.tight_layout()



def save_fig_as_png(figtitle):
    '''
    Saves the current figure into the output folder
    The figtitle should not contain the ".png".
    '''

    fig = plt.gcf()
    path = savedir+figtitle+".png"
    print(path)
    fig.savefig(path, dpi = 300)



def load_spectra(type_sample, name_Sample, name_Substrate):
    '''
    Load the spectra from a txt file. This file must have a first colum with the
    wavelength values and a second column with the transmittance (from 0 to 1). 

    We need to cropped the spectra to the Near-Infrared (NIR) range, 800-2100 nm.
    Note: NIR should go up to 2500 nm. However, because our substrate start to
    absorb around 2100, we need to crop that region

    We will call the function below 'plot_original_NIR' to plot the original and 
    the cropped spectra
    
    args:
        type (string):          'simulated' or 'real' 
        name_spectrum (string):  Name of the spectrum txt file (without .txt)
        name_substrate (string): Name of the substrate txt file (without .txt)

    outputs:
        wv (torch.tensor):  (Cropped) wavelengths (in nanometers, from 800 to 2100)
        T (torch.tensor):   (Cropped) transmittance values (from 0 to 1)
        We also plot the original and the cropped spectra calling another function
    '''

    # STEP 1. Downloading the sample data for real spectra
    if type_sample == 'real':
        
        file_Sample = 'data/Real_samples/'+ name_Sample + '.txt'
        file_Substrate = 'data/Real_samples/'+ name_Substrate + '.txt'

        data = np.loadtxt(file_Sample)             # Sample data
        wv_original = np.round(np.flip(data[:,0])) # Wavelength (in nm)
        T_original = np.flip(data[:,1])/100        # Sample transmittance (range 0-1)

        data_s = np.loadtxt(file_Substrate)        # Substrate alone data
        Ts_original = np.flip(data_s[:,1])/100     # Substrate transmittance (range 0-1)
    
    # STEP 2. From original data to the cropped spectrum (800-2100 nm only)
    wv, T, Ts = cropped_spectrum(wv_original, T_original, Ts_original)
    
    # STEP 3. Plot the original and crop spectra
    plot_original_NIR(wv_original, wv, T_original, T, Ts_original, Ts)
    
    # STEP 4. Convering from numpy to torch tensors (we return the cropped spectrum)
    T[T<0] = 0  # All values should be non-negative
    
    wv_torch = torch.from_numpy(wv)
    T_torch = torch.from_numpy(T)
    Ts_torch = torch.from_numpy(Ts)


    return wv_torch, T_torch, Ts_torch


def cropped_spectrum(wv_original, T_original, Ts_original):
    pos1 = 0
    for i in range(len(wv_original)):
        if wv_original[i]<800:
            pos1 = pos1 + 1

    pos2 = 0
    for i in range(len(wv_original)):
        if wv_original[i]<2100:
            pos2 = pos2 + 1

    wv = wv_original[pos1:pos2]
    T = T_original[pos1:pos2]
    Ts = Ts_original[pos1:pos2]

    return wv, T, Ts


def plot_original_NIR(wv_original, wv, T_original, T, Ts_original, Ts):
    """
    It plots the original spectra (left) and the cropped spectra (right).
    The cropped spectra is in the Near-Infrared (NIR) range 800-2100 nm.
    
    args:
        wv_original (np.narray): Original wavelength (usually between 300-2500 nm)
        wv (np.narray): Cropped wavelength (800-2100 nm)
        T_original (np.narray): Original transmittance of the sample (from 0 to 1)
        T (np.narray): Original transmittance of the sample (from 0 to 1)
        Ts_original (np.narray): Original transmittance of the substrate alone (from 0 to 1)
        Ts (np.narray): Original transmittance of the substrate alone (from 0 to 1)

    outputs:
        void: plots both spectras
    """

    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
    fig.suptitle("Transmission spectrum")
    ax = axes.ravel()

    ax[0].plot(wv_original, T_original)
    ax[0].plot(wv_original, Ts_original)
    ax[0].set_title("Original spectrum")
    ax[0].set_xlabel('Wavelength (nm)')
    ax[0].set_ylabel('Transmittance')
    ax[0].set_ylim(0,1)


    ax[1].plot(wv, T)
    ax[1].plot(wv, Ts)
    ax[1].set_title("NIR cropped spectrum")
    ax[1].set_xlabel('Wavelength (nm)')
    ax[1].set_ylabel('Transmittance')
    ax[1].set_ylim(0,1)

    plt.ion()
    plt.ioff()

