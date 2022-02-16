# Basic libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional
import scipy.optimize

class models():
    def __init__(self, wavelength, transmission, T_substrate, model_n, model_k):
        '''
        Explanation: Tensors
        '''
        self.wv = wavelength    # Vector with the wavelength we used for measure
        self.T = transmission   # Transmission of whole sample
        self.Ts = T_substrate   # Transmission of substrate alone
        self.model_n = model_n  # Model for Refractive Index n, e.g., 'Cauchy'
        self.model_k = model_k  # Model for Attenuation Coefficient k, e.g., 'Exponential'

        self.s = 1/self.Ts+torch.sqrt(1/self.Ts**2-1)
  
    def ref_ind(self, n_parameters):
        '''
        Explanation: Tensors
        '''
        K = 1240.6; # Constant to convert wavelength (nm) to photon energy (eV)
        
        if self.model_n == 'Wemple-DiDomenico':

            # Parameters
            E0 = n_parameters[0]
            Ed = n_parameters[1]
            
            # Wemple-DiDomenico model
            aux =1 + (E0 * Ed)/(E0**2 - (K/self.wv)**2)
            torch.nn.functional.relu(aux)
            n = torch.sqrt(aux)

        return n 

    def ext_coeff(self, k_parameters):
        '''
        Explanation: Tensors
        '''

        if self.model_k == 'Urbach':

            # Parameters
            alpha0 = k_parameters[0]*10**(-6)
            Eu = k_parameters[1]

            lamb0 = (1/8065.5439)*10**(7)   # Wavelength (cm) for 1 eV (passed to nm)
            E = torch.flip(lamb0/self.wv, dims=[0])  # Conversion wavelength to eV
        
            # Urbach model
            alpha = torch.flip(alpha0*torch.exp(E/Eu), dims=[0])  
            k = alpha*self.wv/(4*torch.pi)

        return k 

    
    
    def boundaries_nonKK(self):
        '''
        Explanation: Tensors
        '''

        bound = torch.zeros((7,2)) # For each of the 5 variables, we have the lower and upper boundary
        initial_point = torch.zeros((7))

        if self.model_n == "Wemple-DiDomenico":
            bound[0,0]=2.5; bound[0,1]=4.0; initial_point[0]=3.7
            bound[1,0]=25; bound[1,1]=40; initial_point[1]=33
            bound[2,0]=1; bound[2,1]=1+1e-7; initial_point[2]=1

        if self.model_k == "Urbach":
            bound[3,0]=1.0 ; bound[3,1]=3; initial_point[3]=2.237
            bound[4,0]=0.22; bound[4,1]=0.29; initial_point[4]=0.247

        # For the film thickness and wedge
        bound[5,0]=1000; bound[5,1]=1500; initial_point[5]=1000
        bound[6,0]=0; bound[6,1]=50; initial_point[6]=0

        return bound, initial_point


    
    def generate_spectra(self, parameters):
        '''
        Explanation: Tensors
        '''

        # Variables/parameters that we are looking for:
        n_param = parameters[0:3]
        k_param = parameters[3:5]
        thickness = parameters[5].item()
        wedge = parameters[6].item()
        
        
        RI_n = self.ref_ind(n_param)
        EC_k = self.ext_coeff(k_param)
        AC_alpha = 4*np.pi*EC_k/self.wv

        A=16*(RI_n**2)*self.s
        B=((RI_n+1)**3)*(RI_n+self.s**2)
        C=2*(RI_n**2-1)*(RI_n**2-self.s**2)
        D=((RI_n-1)**3)*(RI_n-self.s**2)

        phi=4*np.pi*RI_n*thickness/self.wv
        x=np.exp(-AC_alpha*thickness)

        T=(A*x)/(B-C*x*np.cos(phi)+D*(x**2)) # Transmission spectra
        T_min=(A*x)/(B+C*x+D*(x**2))
        T_max=(A*x)/(B-C*x+D*(x**2))
        T_aver=np.sqrt(T_min*T_max)

        return T, T_max