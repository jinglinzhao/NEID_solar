import numpy as np
import os
import glob
import pandas as pd
from astropy.io import fits
from datetime import timedelta, date
from datetime import datetime
from alive_progress import alive_bar
import matplotlib.pyplot as plt

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
filenames   = [quality_df['Filename'][i][-27:-5] for i in range(len(quality_df))]
path_prefix = '/gpfs/group/ebf11/default/pipeline/data/neid_solar/v1.1/outputs/jvz5625/'

# parameters 
start_date  = date(2020, 6, 1)
end_date    = date(2020, 6, 30)
plot        = False

CCF, σ_CCF  = [], []
bjd, rv, σrv = np.array([]), np.array([]), np.array([])

# start_time  = datetime.now()
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
                ccf_nor     = ccf_per_obs[idx] / np.median(ccf_per_obs[~idx])
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
                rv      = np.append(rv, df['rv_drp'])
                σrv     = np.append(σrv, df['σrv_drp'])

            bar()

plt.plot(v_grid[idx], (1-CCF))
plt.show()

df, shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(v_grid[idx], (1-CCF).T, σ_CCF.T)











