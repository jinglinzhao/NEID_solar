import numpy as np
from scipy.optimize import minimize
from datetime import timedelta, date
from datetime import datetime
from scipy import stats
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import george
from george import kernels
import celerite
from celerite import terms


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days +1)):
        yield start_date + timedelta(n)


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if np.all(weights==0):
        average = 0
        variance = 0
    else:
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def weighted_avg_1D(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # variance = np.average((values-average)**2, weights=weights)
    # return (average, np.sqrt(variance))
    return average


def weighted_avg_2D(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=1)
    # Fast and numerically precise:
    # variance = np.average(((np.subtract(values.T,average))**2).T, weights=weights, axis=1)
    # return (average, np.sqrt(variance))
    return average


def random_ts(t, y, yerr, test_ratio=0.2):
    '''
        Randomly select; 
        Not suitable for time series data; 
        Not used in the following.
    '''
    idx_test = sorted(random.sample(list(np.arange(len(t))), int(len(t)*test_ratio)))
    idx_train = [n for n in np.arange(len(t)) if not (n in idx_test)]
    t_train, y_train, yerr_train = t[idx_train], y[idx_train], yerr[idx_train]
    t_test, y_test, yerr_test = t[idx_test], y[idx_test], yerr[idx_test]
    return idx_test, t_train, y_train, yerr_train, t_test, y_test, yerr_test

def sample_ts(t, y, yerr, n_splits, test_portion):
    
    from sklearn.model_selection import KFold
    '''
        Return a consecutive subsample specified by test_portion.
    '''
    i = -1
    kf = KFold(n_splits)
    for train_index, test_index in kf.split(np.arange(len(t))):
        i+=1
        if (i==test_portion):
            t_train, y_train, yerr_train = t[train_index], y[train_index], yerr[train_index]
            t_test, y_test, yerr_test = t[test_index], y[test_index], yerr[test_index]
            return t_train, y_train, yerr_train, t_test, y_test, yerr_test                
        
        
def moving_ave(t, y, width=2):
    w_aves = []
    for t_i in t:
        weights = stats.norm.pdf(t, t_i, width)
        # weights = np.multiply(abs(array_x-t_i)<width, 1) 
        w_ave, _ = weighted_avg_and_std(y, weights)
        w_aves.append(w_ave)
        # w_stds.append(w_std)
    return np.array(w_aves)
        
        
def corr(t, y):
    spacing = []
    diff = []
    for i in range(len(t)):
        spacing = np.append(spacing, t[i:]-t[i])
        diff = np.append(diff, y[i:]-y[i])
    return spacing, diff

def moving_std(t, array_x, array_y, width):
    w_aves = []
    w_stds = []
    for t_i in t:
        # weights = stats.norm.pdf(array_x, t_i, width)
        weights = np.multiply(abs(array_x-t_i)<width, 1) 
        w_ave, w_std = weighted_avg_and_std(array_y, weights)
        w_aves.append(w_ave)
        w_stds.append(w_std)
    return np.array(w_aves), np.array(w_stds)

def cov(x, y):
    # return sum((x-np.mean(x))*(y-np.mean(y)))/(len(x)-1)
    return sum((x-np.mean(x))*(y-np.mean(y)))/len(x)


def plot_ts_corr(k_mode, t, rv, erv, ind, eind, ts_xlabel, rv_xlabel, pe_xlabel, ind_yalbel, file_name, height_ratio=0.7, vlines=[], HARPS=True):

	'''
	e.g. 
		k_mode 		= 11
		t 			= bjd_daily
		rv 			= rv_daily
		erv 		= erv_daily
		ind 		= ΔRV_k
		eind 	 	= eRV_FT_k
		ts_xlabel 	= 'BJD - 2400000 [d]'
		rv_xlabel 	= '$RV_{HARPS}$'
		pe_xlabel 	= 'Period [days]'
		ind_yalbel	= 'A'
		file_name 	= 'time-series_and_shift_correlation.pdf'

	'''

	from sklearn.linear_model import LinearRegression

	# set up the plotting configureations
	alpha1, alpha2 = [0.5,0.2]
	widths 	= [7,1]
	heights = [1]*(k_mode+1)
	gs_kw 	= dict(width_ratios=widths, height_ratios=heights)
	plt.rcParams.update({'font.size': 12})
	fig6, f6_axes = plt.subplots(figsize=(16, k_mode+1), ncols=2, nrows=k_mode+1, constrained_layout=True,
	                             gridspec_kw=gs_kw)

	# plots 
	for r, row in enumerate(f6_axes):
		for c, ax in enumerate(row):	

			# time-series 
			if c==0:
				if r==0:
					ax.errorbar(t, rv-np.mean(rv), erv, marker='.', ms=5, color='black', ls='none', alpha=alpha1)
					ax.set_title('Time-series')
					ax.set_ylabel(rv_xlabel)
				else:				
					ax.errorbar(t, ind[r-1,:], eind[r-1,:],  marker='.', ms=5, color='black', ls='none', alpha=alpha1)
					ax.set_ylabel(ind_yalbel + '$_{' + str(r) + '}$')
				if r!=k_mode:
					ax.tick_params(labelbottom=False)
				else:
					ax.set_xlabel(ts_xlabel)

			if c==1:
				if r==0:
					reg = LinearRegression().fit(rv.reshape(-1, 1), rv.reshape(-1, 1))
					score = reg.score(rv.reshape(-1, 1), rv.reshape(-1, 1))
					adjust_R2 = 1-(1-score)*(len(t)-1)/(len(t)-1-1)
					title = r'$\bar{R}$' + r'$^2$'
					ax.set_title(title + ' = {:.2f}'.format(adjust_R2))					
					ax.plot(rv-np.mean(rv), rv-np.mean(rv), 'k.', alpha = alpha2)				
				if r>0:
					reg = LinearRegression().fit(rv.reshape(-1, 1), ind[r-1,:].reshape(-1, 1))
					score = reg.score(rv.reshape(-1, 1), ind[r-1,:].reshape(-1, 1))
					adjust_R2 = 1-(1-score)*(len(t)-1)/(len(t)-1-1)
					title = r'$\bar{R}$' + r'$^2$'
					ax.set_title(title + ' = {:.2f}'.format(adjust_R2))
					ax.plot(rv-np.mean(rv), ind[r-1,:], 'k.', alpha = alpha2)
				if r!=k_mode:
					ax.tick_params(labelbottom=False)
				else:
					ax.set_xlabel(rv_xlabel)
				ax.yaxis.tick_right()

	fig6.align_ylabels(f6_axes[:, 0])
	if file_name !=[]:
		plt.savefig(file_name)
	plt.show()    
	plt.close('all')
    
    
# def GP(t, y, yerr):
#     w0 = 2*np.pi/5.2
#     Q = 10
#     S0 = np.var(y) / (w0 * Q)
#     bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
#     kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0), bounds=bounds)
#     # kernel.freeze_parameter("log_omega0")

#     gp = celerite.GP(kernel, mean=np.mean(y))
#     gp.compute(t, yerr)  # You always need to call compute once.

#     initial_params = gp.get_parameter_vector()
#     bounds = gp.get_parameter_bounds()

#     r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
#     gp.set_parameter_vector(r.x)
    
#     return gp


# def GP_fit_Matern52Kernel(x, yerr, r):
#     kernel 	= kernels.Matern52Kernel(r**2)
#     gp 		= george.GP(kernel)
#     gp.compute(x, yerr)
#     return gp


# def GP_fit_p1(t, y, yerr, p):
#     w0 = 2*np.pi/p
#     Q = 6
#     S0 = np.var(y) / (w0 * Q)
#     bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
#     kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0), bounds=bounds)
#     kernel.freeze_parameter("log_omega0")
#     kernel.freeze_parameter("log_Q")

#     gp = celerite.GP(kernel, mean=np.mean(y))
#     gp.compute(t, yerr)  # You always need to call compute once.

#     initial_params = gp.get_parameter_vector()
#     bounds = gp.get_parameter_bounds()

#     r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
#     gp.set_parameter_vector(r.x)
    
#     return gp


# def GP_fit_Q1(t, y, yerr, Q):
#     w0 = 2*np.pi/5
#     S0 = np.var(y) / (w0 * Q)
#     bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(np.log(2*np.pi/7), np.log(2*np.pi/3)))
#     kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0), bounds=bounds)
#     # kernel.freeze_parameter("log_omega0")
#     kernel.freeze_parameter("log_Q")

#     gp = celerite.GP(kernel, mean=np.mean(y))
#     gp.compute(t, yerr)  # You always need to call compute once.

#     initial_params = gp.get_parameter_vector()
#     bounds = gp.get_parameter_bounds()

#     r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
#     gp.set_parameter_vector(r.x)
    
#     return gp