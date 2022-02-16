# Basic libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional
import scipy.optimize
from src.models import * 

class generating():
    def __init__(self, wavelength, T_substrate, model_n, model_k):
        '''
        Explanation: Tensors
        '''
        self.wv = wavelength    # Vector with the wavelength we used for measure
        self.Ts = T_substrate   # Transmission of substrate alone
        self.model_n = model_n  # Model for Refractive Index n, e.g., 'Cauchy'
        self.model_k = model_k  # Model for Attenuation Coefficient k, e.g., 'Exponential'

        self.s = 1/self.Ts+torch.sqrt(1/self.Ts**2-1)

    
    def spectra(self, n_samples):
        '''
        Explanation: Tensors
        '''

        # Bounds for the optical parameters in n (refractive index) and k (extinction coefficient)
        model = models(wavelength = self.wv, T_substrate = self.Ts, \
                model_n = self.model_n, model_k = self.model_n)

        bound, initial_point = model.boundaries_nonKK()
        
        # Number of optical parameters, and number of features
        # There are 3 parameters for n, 2 for k, the thickness and wedge
        n_parameters = 7  
        n_features = len(self.wv)

        # Input: Transmission spectrum {T(lambda)} = {0.1, 0.15, 0.2, 0.1, ...}
        # Output: Envelope of the spectrum {T_max(lambda)} = {0.6, 0.61, 0.62, 0.65, ...}
        input = torch.zeros((n_samples, n_features)) 
        output = torch.zeros((n_samples, n_features))  

        # Loop to create the dataset
        torch.manual_seed(0) 

        for i in range(n_sample):
            p = torch.zeros((7))
            for k in range(n_parameters-1): # Last parameter for wedge is set for 0 so far
                p[k] = torch.FloatTensor(1).uniform_(bound[k,0].item(), bound[k,1].item())
            input[i], output[i] = model.generate_spectra(p)

        return input, output