import numpy as np
import os
import glob
import pandas as pd
from astropy.io import fits
from datetime import timedelta, date
from datetime import datetime
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
# start_date  = date(2020, 5, 26)
# end_date    = date(2020, 6, 25)
# start_date  = date(2020, 6, 23)
# end_date    = date(2020, 6, 23)
start_date  = date(2020, 1, 1)
end_date    = date(2020, 12, 31)

plot        = False
o_start     = 55    # 56 in Julia (physical 118)
o_end       = 108   # 108 in julia (physical 65)
# o_exclude   = np.array([61, 66, 81, 90]) -> version 1 (data_v1)
# o_exclude   = 173 - np.array([66, 67, 68, 74, 76, 83, 118]) # -> version 2 (data); index starting from 0
# o_exclude   = 173 - np.array([66, 67, 68, 69, 73, 74, 75, 76, 83, 118]) # -> version 3 (data_v3); index starting from 0
o_exclude   = 173 - np.array([66, 67, 68, 69, 73, 74, 75, 76, 78, 83, 107,108,112,113,117, 118]) # -> version 4 (data_v4);
o_used      = np.array([x for x in np.arange(o_start, o_end) if (x in o_exclude) == False])

v_grid      = -100 + np.arange(1604)*0.25
idx_v       = (v_grid>87) & (v_grid<111)

CCF, σCCF                       = [], []
bjd, rv, σrv                    = np.array([]), np.array([]), np.array([])
CCF_daily, σCCF_daily           = [], []
bjd_daily, rv_daily, σrv_daily  = np.array([]), np.array([]), np.array([])

start_time  = datetime.now()
for single_date in daterange(start_date, end_date):

    print(single_date.strftime("%Y-%m-%d"))

    path        = path_prefix + single_date.strftime('extracted_ccf/%m/%d/')
    file_ccf    = sorted(glob.glob(path + '/*.ccf')) 
    N_file      = len(file_ccf)

    # path_save   = path_prefix + single_date.strftime('ccf_by_obs_56_108/%m/%d/')
    # if not os.path.exists(path_save):
    #     os.makedirs(path_save)

    if N_file != 0:

        with alive_bar(N_file) as bar:

            for n in range(N_file):
                ccf_per_order   = np.loadtxt(file_ccf[n])

                for order in o_used:
                    if ccf_per_order[order, :].all() == 0:
                        continue
                    else:
                        reg                     = LinearRegression().fit(v_grid[~idx_v].reshape(-1,1), ccf_per_order[order, ~idx_v])
                        fitted_continuum        = reg.predict(v_grid.reshape(-1,1))
                        ccf_per_order[order, :] = ccf_per_order[order, :] / fitted_continuum * np.median(fitted_continuum)

                ccf_per_obs   = np.sum(ccf_per_order[o_used, :], axis=0)

                # np.savetxt(path_save + file_ccf[n][-27:-4] + '.ccf', ccf_per_obs)

                if not np.any(CCF):
                    CCF = ccf_per_obs #(1604,)
                else:
                    CCF = np.vstack((CCF, ccf_per_obs)) # e.g.(53, 1604)

                df      = quality_df[quality_df['Filename'].str.contains(file_ccf[n][-26:-4])]
                bjd     = np.append(bjd, df['jd_drp'])
                rv      = np.append(rv, df['rv_drp']*1000)
                σrv     = np.append(σrv, df['σrv_drp']*1000)

                bar()

                if (n==0) & (plot==True): # only plot once
                
                    ccf_per_order_reject = np.vstack((ccf_per_order[0:o_start, :], ccf_per_order[o_end:, :]))
                    continuum   = np.mean(ccf_per_order_reject, axis=1)
                    idx         = (continuum!=0)
                    plt.plot(v_grid, ccf_per_order_reject[idx,:].T/continuum[idx], 'r', alpha=0.1)

                    continuum   = np.mean(ccf_per_order[o_used, :], axis=1)
                    idx         = (continuum!=0)
                    plt.plot(v_grid, ccf_per_order[o_used, :][idx,:].T/continuum[idx], 'b', alpha=0.1)
                    plt.show()

                    for i in range(ccf_per_order.shape[0]):
                        plt.plot(v_grid, ccf_per_order[i,:])
                        plt.title('Julia index' + str(i+1) + ' / order ' + str(174-i-1))
                        plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

#----------------------------------
# Save data
#----------------------------------
σCCF    = CCF[:,idx_v].T**0.5 / np.median(CCF[:,~idx_v], axis=1)
CCF     = 1 - CCF[:,idx_v].T / np.median(CCF[:,~idx_v], axis=1)     # normalisation 

if 0: 
    np.savetxt('./data/v_grid.txt', v_grid)
    np.savetxt('./data/CCF.txt', CCF)
    np.savetxt('./data/σCCF.txt', σCCF)
    np.savetxt('./data/bjd.txt', bjd)
    np.savetxt('./data/rv.txt', rv)
    np.savetxt('./data/σrv.txt', σrv)

if 1:
    np.savetxt('./data_v4/v_grid.txt', v_grid[idx_v])
    np.savetxt('./data_v4/CCF.txt', CCF)
    np.savetxt('./data_v4/σCCF.txt', σCCF)
    np.savetxt('./data_v4/bjd.txt', bjd)
    np.savetxt('./data_v4/rv.txt', rv)
    np.savetxt('./data_v4/σrv.txt', σrv)

plt.plot(v_grid[idx_v], CCF)
plt.show()

plt.close('all')