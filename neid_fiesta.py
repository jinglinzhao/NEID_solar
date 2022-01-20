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
filenames   = [quality_df['Filename'][i][-27:] for i in range(len(quality_df))]
path_prefix = '/gpfs/group/ebf11/default/pipeline/data/neid_solar/v1.1/outputs/jvz5625/'

# parameters 
start_date  = date(2020, 1, 1)
end_date    = date(2020, 12, 31)
plot        = False
o_start     = 55    # 56 in Julia
o_end       = 108   # 108 in Julia

start_time  = datetime.now()
for single_date in daterange(start_date, end_date):

    print(single_date.strftime("%Y-%m-%d"))

    path        = path_prefix + single_date.strftime('extracted_ccf/%m/%d/')
    file_ccf    = sorted(glob.glob(path + '/*.ccf')) 
    N_file      = len(file_ccf)

    path_save   = path_prefix + single_date.strftime('ccf_by_obs_56_108/%m/%d/')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if N_file != 0:
        CCF = []
        with alive_bar(N_file) as bar:
            for n in range(N_file):
                ccf_per_order = np.loadtxt(file_ccf[n])
                ccf_per_obs   = np.sum(ccf_per_order[o_start:o_end, :], axis=0)
                np.savetxt(path_save + file_ccf[n][-27:-4] + '.ccf', ccf_per_obs)
                if not np.any(CCF):
                    CCF = ccf_per_obs
                else:
                    CCF = np.vstack((CCF, ccf_per_obs)) 
                bar()

                if plot == True:
                    v_grid  = -100 + np.arange(len(ccf_per_obs))*0.25
                
                    continuum   = np.mean(ccf_per_order[o_start:o_end, :], axis=1)
                    idx         = (continuum!=0)
                    plt.plot(v_grid, ccf_per_order[o_start:o_end, :][idx,:].T/continuum[idx], 'b', alpha=0.5)

                    ccf_per_order_reject = np.vstack((ccf_per_order[0:o_start, :], ccf_per_order[o_end:, :]))
                    continuum   = np.mean(ccf_per_order_reject, axis=1)
                    idx         = (continuum!=0)
                    plt.plot(v_grid, ccf_per_order_reject[idx,:].T/continuum[idx], 'r', alpha=0.5)

                    # plt.xlim(90,110)
                    # plt.savefig(single_date.strftime('./normalised_ccf_by_order/%m-%d.png'))
                    plt.show()

        np.savetxt(path_prefix + single_date.strftime("ccf_by_day_56_108/%Y-%m-%d.CCF"), CCF)
