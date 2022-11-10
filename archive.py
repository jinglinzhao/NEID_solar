import numpy as np
import os
import glob
import pandas as pd
from astropy.io import fits
from datetime import timedelta, date
from datetime import datetime
from alive_progress import alive_bar
import matplotlib.pyplot as plt

from FIESTA_functions import *
from HARPS_N_functions import *

#--------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days +1)):
        yield start_date + timedelta(n)

#--------------------------------------------------------------------
# Read CCFs
#--------------------------------------------------------------------
quality_df  = pd.read_csv('combined_rvs_1.csv')

# order 66 to 118
rv_by_order = quality_df.iloc[:,75::2].values
σrv_by_order = quality_df.iloc[:,76::2].values

filenames   = [quality_df['Filename'][i][-27:-5] for i in range(len(quality_df))]
path_prefix = '/gpfs/group/ebf11/default/pipeline/data/neid_solar/v1.1/outputs/jvz5625/'

# parameters 
# start_date  = date(2020, 5, 26)
# end_date    = date(2020, 6, 25)
# start_date  = date(2020, 6, 23)
# end_date    = date(2020, 6, 23)
start_date  = date(2020, 1, 1)
end_date    = date(2020, 12, 31)

CCF, σ_CCF  = [], []
bjd, rv, σrv = np.array([]), np.array([]), np.array([])

start_time  = datetime.now()
for single_date in daterange(start_date, end_date):

    print(single_date.strftime("%Y-%m-%d"))

    path_read   = path_prefix + single_date.strftime('ccf_by_obs_56_108/%m/%d/')
    file_ccf    = sorted(glob.glob(path_read + '/*.ccf')) 
    N_file      = len(file_ccf)

    if N_file != 0:
        with alive_bar(N_file) as bar:
            for n in range(N_file):
                ccf_per_obs = np.loadtxt(file_ccf[n])
                v_grid      = -100 + np.arange(len(ccf_per_obs))*0.25
                idx         = (v_grid>85) & (v_grid<113)
                ccf_nor     = ccf_per_obs[idx]      / np.median(ccf_per_obs[~idx])
                σ_ccf_nor   = ccf_per_obs[idx]**0.5 / np.median(ccf_per_obs[~idx])
                # plt.plot(v_grid[idx], ccf_nor)
                # plt.show()                
                if not np.any(CCF):
                    CCF     = ccf_nor
                    σ_CCF   = σ_ccf_nor
                else:
                    CCF     = np.vstack((CCF, ccf_nor)) 
                    σ_CCF   = np.vstack((σ_CCF, σ_ccf_nor)) 

                df      = quality_df[quality_df['Filename'].str.contains(file_ccf[n][-26:-4])]
                bjd     = np.append(bjd, df['jd_drp'])
                rv      = np.append(rv, df['rv_drp']*1000)
                σrv     = np.append(σrv, df['σrv_drp']*1000)

                bar()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

v_grid  = v_grid[idx]
CCF     = (1-CCF).T
σ_CCF   = σ_CCF.T

#----------------------------------
# Save data
#----------------------------------
if 1:
    np.savetxt('./data/v_grid.txt', v_grid)
    np.savetxt('./data/CCF.txt', CCF)
    np.savetxt('./data/σ_CCF.txt', σ_CCF)
    np.savetxt('./data/bjd.txt', bjd)
    np.savetxt('./data/rv.txt', rv)
    np.savetxt('./data/σrv.txt', σrv)

if 0: 
    np.savetxt('./data_526_625/v_grid.txt', v_grid)
    np.savetxt('./data_526_625/CCF.txt', CCF)
    np.savetxt('./data_526_625/σ_CCF.txt', σ_CCF)
    np.savetxt('./data_526_625/bjd.txt', bjd)
    np.savetxt('./data_526_625/rv.txt', rv)
    np.savetxt('./data_526_625/σrv.txt', σrv)
    



    
    
def ts_statistics(t, y, yerr):
    '''
        refer to Timeseries Statistics
    '''
    gp = GP(t, y, yerr)

    ###############
    # time series #
    ###############
    fig = plt.figure(figsize=(18, 3))
    plt.subplots_adjust(right=0.85, hspace=0.3)
    plt.rcParams.update({'font.size': 14})

    ## Upper panel    
    # plt.title(str(date)[:10])
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    x = np.linspace(min(t), max(t), 1000)
    pred_mean, pred_var = GP(t, y, yerr).predict(y, x, return_var=True)
    pred_std = np.sqrt(pred_var)        
    plt.plot(x, pred_mean, color='r', alpha=0.3)
    plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color='r', alpha=0.1,
                     edgecolor="none")
    plt.xlabel("Time [min]")
    plt.ylabel("RV [m/s]")

    ## Correlation statistics
    fig = plt.figure(figsize=(18, 3))
    spacing, diff = corr(t, y)
    plt.plot(spacing, diff, 'k.', alpha=0.3)        
    plt.xlabel(r'$\Delta$T [min]')
    plt.ylabel(r'$\Delta$RV [m/s]')

    plt.show()

    
    w_ave, w_std = moving_std(spacing, diff, width=0.02)
    
    ###################
    # Plot statistics #
    ###################
    fig = plt.figure(figsize=(18, 3))
    plt.title('all data')
    plt.plot(spacing, w_std, 'r.', alpha=0.5)
    plt.xlabel(r'$\Delta$T [min]')
    plt.ylabel('std [m/s]')
    # plt.plot(spacing, w_ave, 'b.', alpha=0.5)
    plt.show()

    fig = plt.figure(figsize=(18, 3))
    plt.title('zoom in')
    plt.plot(spacing, diff, 'k.')
    plt.plot(spacing, w_std, 'r.')
    plt.xlabel(r'$\Delta$T [min]')
    plt.ylabel('std [m/s]')
    plt.xlim(1.5,1.6)

    fig = plt.figure(figsize=(18, 3))
    plt.title('zoom out')
    plt.plot(spacing, diff, 'k.')
    plt.plot(spacing, w_std, 'r.')
    plt.xlabel(r'$\Delta$T [min]')
    plt.ylabel('std [m/s]')
    # plt.xlim(100, 110)
    plt.xlim(10, 20)
    plt.show()
    
    ###############
    # Periodogram #
    ###############
    from astropy.timeseries import LombScargle

    plt.subplots(figsize=(15, 3))
    plt.rcParams.update({'font.size': 14})
    frequency, power = LombScargle(spacing, w_std).autopower(minimum_frequency=0.1, maximum_frequency=1, samples_per_peak=10)        
    plt.plot(1/frequency, power, 'r', alpha=0.5, label='corr std')
    frequency, power = LombScargle(t, y).autopower(minimum_frequency=0.1, maximum_frequency=1, samples_per_peak=10)
    plt.plot(1/frequency, power, 'k', alpha=0.5, label='time series')
    plt.ylabel('Power')
    plt.xlabel('Period [min]')
    plt.legend()
    plt.show()

    plt.subplots(figsize=(15, 3))
    plt.rcParams.update({'font.size': 14})
    frequency, power = LombScargle(spacing, w_std).autopower(minimum_frequency=0.005, maximum_frequency=1, samples_per_peak=10)        
    plt.plot(1/frequency, power, 'r', alpha=0.5, label='corr std')
    frequency, power = LombScargle(t, y).autopower(minimum_frequency=0.1, maximum_frequency=1, samples_per_peak=10)
    plt.plot(1/frequency, power, 'k', alpha=0.5, label='time series')
    plt.ylabel('Power')
    plt.xlabel('Period [min]')
    plt.legend()
    plt.xscale('log')
    plt.show()    