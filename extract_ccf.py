import numpy as np
import os
import glob
import pandas as pd
from astropy.io import fits
from datetime import timedelta, date
from datetime import datetime

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

start_date  = date(2020, 6, 1)
end_date    = date(2020, 6, 1)

start_time  = datetime.now()
for date in daterange(start_date, end_date):

    print(date.strftime("%Y-%m-%d"))
    path = ('./data/' + date.strftime('%m') + '/' + date.strftime('%d'))
    if not os.path.exists(path):
        os.makedirs(path)
        print('The new directory ' + path + ' is created!')

    file_ccf = sorted(glob.glob('/gpfs/group/ebf11/default/pipeline/data/neid_solar/v1.1/L2/2021/' + \
                                date.strftime('%m') + '/' + date.strftime('%d') + '/*.fits')) 

    N_file		= len(file_ccf)
    bjd 		= np.zeros(N_file)
    rv 			= np.zeros(N_file)
    σrv			= np.zeros(N_file)    
    CCF 		= []

    for n in range(N_file):
        if file_ccf[n][-27:] in filenames:
            with fits.open(file_ccf[n]) as hdulist:
                header 	= hdulist[12].header
                bjd[n] 	= header['CCFJDMOD']
                rv[n] 	= header['CCFRVMOD']*1000
                σrv[n]	= header['DVRMSMOD']*1000
                ccf_per_order   = hdulist[12].data                
                ccf_per_obs     = np.sum(ccf_per_order, axis=0)
                v_grid 	= header['CCFSTART'] + np.arange(len(ccf_per_obs))*header['CCFSTEP']
                if n == 0:
                    CCF = ccf_per_obs
                else:
                    CCF = np.vstack((CCF, ccf_per_obs)) 

            np.savetxt(path + '/' + file_ccf[n][-27:-4] + 'ccf', ccf_per_order)
            np.savetxt('./data/' + date.strftime("%Y-%m-%d") + '.CCF', CCF)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

idx     = bjd!=0
bjd     = bjd[idx]
rv      = rv[idx]
σrv     = σrv[idx]
CCF     = CCF       #(N_file, N_v)
N_file  = len(bjd)  # update N_file