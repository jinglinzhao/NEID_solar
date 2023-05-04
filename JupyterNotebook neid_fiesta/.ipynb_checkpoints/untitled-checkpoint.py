import numpy as np
from scipy import optimize as opt

def wmse(y1, y2, w):
    wmse = np.average((y1-y2)**2, weights = w)
    return wmse

def model_1(params, t, X_dim, X):
    planet_rv_model = 10**params[X_dim+1]*np.sin(10**params[X_dim+3]*t + params[X_dim+2]) 
    stellar_rv_model = np.sum(params[:X_dim]*X, axis=1) + params[X_dim]
    rv_model = planet_rv_model + stellar_rv_model
    return rv_model

def model_2(params, t):
    '''
        Keplerian model
    '''
    planet_rv_model = 10**params[0]*np.sin(10**params[2]*t + params[1]) 
    return planet_rv_model



def loss_1(params, Y, X, W, t, X_dim, amp, t_inj, phase_inj):
    y = Y + amp*np.sin(t_inj*t+phase_inj)  
    return wmse(model_1(params, t, X_dim, X), y, W)

def loss_2(params, Y, X, W, t, X_dim, amp, t_inj, phase_inj):
    y = Y + amp*np.sin(t_inj*t+phase_inj)  
    return wmse(model_2(params, t), y, W)

        
# def execute(args):
#     i, Y, X, W, t, X_dim, amp, t_inj, phase_inj = args
    
def execute(i, Y, X, W, t, X_dim, amp, t_inj, phase_inj):
    
    def loss_1_wrapper(params):
        return loss_1(params, Y, X, W, t, X_dim, amp[i], t_inj, phase_inj)
    
    def loss_2_wrapper(params):
        return loss_2(params, Y, X, W, t, X_dim, amp[i], t_inj, phase_inj)
 
    
    sol_1 = opt.dual_annealing(loss_1, bounds=[[-5,5] for i in np.arange(X_dim+1)] + [[-1.1,1.1], [0, np.pi], [np.log10(t_inj/1.5), np.log10(t_inj*1.5)]],  maxiter=1000)
    sol_2 = opt.dual_annealing(loss_2, bounds= [[-1.1,1.1], [0, np.pi], [np.log10(t_inj/1.5), np.log10(t_inj*1.5)]],  maxiter=1000)
    
    fitted_params_1 = sol_1.x
    fitted_params_2 = sol_2.x    

    w_array_1[i] = fitted_params_1[:X_dim]
    amp_pred_1[i] = 10**fitted_params_1[X_dim+1]
    phase_pred_1[i] = fitted_params_1[X_dim+2]
    t_pred_1[i] = 10**fitted_params_1[X_dim+3]
    rms_array_1[i] = loss_1(fitted_params_1, Y, X, W, t, X_dim, amp[i], t_inj, phase_inj)**0.5

    amp_pred_2[i] = 10**fitted_params_2[0]
    phase_pred_2[i] = fitted_params_2[1]
    t_pred_2[i] = 10**fitted_params_2[2]
    rms_array_2[i] = loss_2(fitted_params_2, Y, X, W, t, X_dim, amp[i], t_inj, phase_inj)**0.5
    
    return w_array_1[i], amp_pred_1[i], phase_pred_1[i], t_pred_1[i] , rms_array_1[i], amp_pred_2[i], phase_pred_2[i], t_pred_2[i], rms_array_2[i]
