# Basic libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import * 

class learning():
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
        self.hey = 2
    
    
    def LSE(self, parameters):
        '''
        Explanation: Tensors
        '''
        model = models(wavelength = self.wv, transmission = self.T, T_substrate = self.Ts, \
            model_n = self.model_n, model_k = self.model_k)

        T_simulated = model.generate_spectra(parameters)*100
        error = torch.sum((self.T*100-T_simulated)**2)/len(self.wv) # In percentage
        return error.item()


    def random_generation(self, fix, n_iters, seed):
        '''
        Explanation: Tensors
        '''
        # We first set the random parameters to zero
        w_random = torch.zeros((7,n_iters))
        torch.manual_seed(seed)     

        # We find the boundaries for each parameters
        model = models(wavelength = self.wv, transmission = self.T, T_substrate = self.Ts, \
            model_n = self.model_n, model_k = self.model_k)

        bound, initial_point = model.boundaries_nonKK()

        # Random parameters [0, 1, 2] for Refractive Index, acording to the boundaries
        if fix[0]<0:
            rand_0 = torch.rand(14)
            w_random[0,:] = rand_0[torch.randint(low=0,high=len(rand_0),size=(n_iters,))]*(bound[0,1]-bound[0,0])+bound[0,0]
        else:
            w_random[0,:] = fix[0]

        if fix[1]<0:
            rand_0 = torch.rand(14)
            w_random[1,:] = rand_0[torch.randint(low=0,high=len(rand_0),size=(n_iters,))]*(bound[1,1]-bound[1,0])+bound[1,0]
        else:
            w_random[1,:] = fix[1]

        if fix[2]<0:
            rand_0 = torch.rand(7)
            w_random[2,:] = rand_0[torch.randint(low=0,high=len(rand_0),size=(n_iters,))]*(bound[2,1]-bound[2,0])+bound[2,0]
        else:
            w_random[2,:] = fix[2]

        # Random parameters [3, 4] for Exctintion Coefficient (absorption), acording to the boundaries
        if fix[3]<0:
            rand_0 = torch.rand(4)
            w_random[3,:] = rand_0[torch.randint(low=0,high=len(rand_0),size=(n_iters,))]*(bound[3,1]-bound[3,0])+bound[3,0]
        else:
            w_random[3,:] = fix[3]

        if fix[4]<0:
            rand_0 = torch.rand(4)
            w_random[4,:] = rand_0[torch.randint(low=0,high=len(rand_0),size=(n_iters,))]*(bound[4,1]-bound[4,0])+bound[4,0]
        else:
            w_random[4,:] = fix[4]


        # Random parameters [5, 6] for thickness and wedge, acording to the boundaries (f they are not fix)
        if fix[5]<0:
            rand_0 = torch.rand(30)
            w_random[5,:] = rand_0[torch.randint(low=0,high=len(rand_0),size=(n_iters,))]*(bound[5,1]-bound[5,0])+bound[5,0]
        else:
            w_random[5,:] = fix[5]
        
        if fix[6]<0:
            rand_0 = torch.rand(3)
            w_random[6,:] = rand_0[torch.randint(low=0,high=len(rand_0),size=(n_iters,))]*(bound[6,1]-bound[6,0])+bound[6,0]
        else:
            w_random[6,:] = fix[6]

        return w_random


    def loop(self, fix, n_iters, w_random):
        global_min = w_random[:,0]
        min_value = self.LSE(w_random[:,0])

        for i in range(n_iters):
            aux = self.LSE(w_random[:,i])

            if aux < min_value:
                global_min = w_random[:,i]
                min_value = aux
                print(f'Iterations {i}, cost: {min_value:.3f} %')
        return global_min, min_value


    def iterations(self, fix, n_iters):
        '''
        Explanation: Tensors
        '''

        # STEP 0. Verbose
        print('\n --- GLOBAL OPTIMIZATION ---')
        print('\nWe will do',n_iters,'iterations for the global optimization \
            (modified random search algorithm)\n')
        
        # STEP 1. GENERATING RANDOM NUMBERS FOR THE ITERATIONS (could be another function)
        w_random = self.random_generation(fix, n_iters, 1)


        # STEP 2. LOOKING FOR THE GLOBAL MINIMUM
        global_min, cost = self.loop(fix, n_iters, w_random)
        
        np.set_printoptions(precision=3, suppress=True)
        print('\nGlobal optimization result:', global_min.numpy())

        if cost > 4:    # A cost (Least Squared Error) more than 4% is bad and we should repeat
            print('\nA good global min is not found, we should repeat the algorithm:\n')
            w_random = self.random_generation(fix, n_iters, 2)
            global_min, cost = self.loop(fix, n_iters, w_random)

        return global_min

