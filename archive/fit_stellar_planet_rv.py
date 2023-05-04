import numpy as np
from multiprocessing import Pool
import time
import math
from scipy import optimize
import scipy.optimize as opt


def execute_in_parallel(amp_inj, X, Y, t_inj, phase_inj, k_max):
    
    N = len(amp_inj)
    w_array_1 = np.zeros((N, k_max))
    amp_pred_1 = np.zeros(N)
    rms_array_1 = np.zeros(N)
    phase_pred_1 = np.zeros(N)
    t_pred_1 = np.zeros(N)

    amp_pred_2 = np.zeros(N)
    rms_array_2 = np.zeros(N)
    phase_pred_2 = np.zeros(N)
    t_pred_2 = np.zeros(N)

    def wmse(y1, y2, w):
        wmse = np.average((y1-y2)**2, weights = w)
        return wmse

    def model_1(params):
        planet_rv_model = 10**params[k_max+1]*np.sin(10**params[k_max+3]*t + params[k_max+2]) 
        stellar_rv_model = np.sum(params[:k_max]*X[:,:k_max], axis=1) + params[k_max]
        rv_model = planet_rv_model + stellar_rv_model
        return rv_model

    def model_2(params):
        '''
            Keplerian model
        '''
        planet_rv_model = 10**params[0]*np.sin(10**params[2]*t + params[1]) 
        return planet_rv_model

    def execute(i):
        y = Y + amp_inj[i] *np.sin(t_inj*t+phase_inj)  

        def loss_1(params):
            return wmse(model_1(params), y, W)

        def loss_2(params):
            return wmse(model_2(params), y, W)    

        sol_1 = opt.dual_annealing(loss_1, bounds=[[-5,5] for i in np.arange(k_max+1)] 
                                   + [[-1.1,1.1], [0, np.pi], [np.log10(t_inj/1.5), np.log10(t_inj*1.5)]],  
                                   maxiter=1000)
        sol_2 = opt.dual_annealing(loss_2, bounds= [[-1.1,1.1], 
                                                    [0, np.pi], 
                                                    [np.log10(t_inj/1.5), np.log10(t_inj*1.5)]],
                                   maxiter=1000)

        fitted_params_1 = sol_1.x
        fitted_params_2 = sol_2.x    

        w_array_1[i] = fitted_params_1[:k_max]
        amp_pred_1[i] = 10**fitted_params_1[k_max+1]
        phase_pred_1[i] = fitted_params_1[k_max+2]
        t_pred_1[i] = 10**fitted_params_1[k_max+3]
        rms_array_1[i] = loss_1(fitted_params_1)**0.5

        amp_pred_2[i] = 10**fitted_params_2[0]
        phase_pred_2[i] = fitted_params_2[1]
        t_pred_2[i] = 10**fitted_params_2[2]
        rms_array_2[i] = loss_2(fitted_params_2)**0.5
        
        return w_array_1[i], amp_pred_1[i], phase_pred_1[i], t_pred_1[i] , rms_array_1[i], amp_pred_2[i], phase_pred_2[i], t_pred_2[i], rms_array_2[i]
