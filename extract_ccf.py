import numpy as np
import os
import glob
import pandas as pd
from astropy.io import fits
from datetime import timedelta, date
from datetime import datetime
from alive_progress import alive_bar

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

start_date  = date(2021, 6, 23)
end_date    = date(2021, 6, 23)

start_time  = datetime.now()
for single_date in daterange(start_date, end_date):

    print(single_date.strftime("%Y-%m-%d"))

    path = path_prefix + 'extracted_ccf/' + single_date.strftime('%m') + '/' + single_date.strftime('%d') + '/'
    if not os.path.exists(path):
        os.makedirs(path)
        print('The new directory ' + path + ' is created!')

    file_ccf = sorted(glob.glob('/gpfs/group/ebf11/default/pipeline/data/neid_solar/v1.1/L2/2021/' + \
                                single_date.strftime('%m') + '/' + single_date.strftime('%d') + '/*.fits')) 

    N_file		= len(file_ccf)
    bjd 		= np.zeros(N_file)
    rv 			= np.zeros(N_file)
    σrv			= np.zeros(N_file)    
    CCF 		= []

    with alive_bar(N_file) as bar:
        for n in range(N_file):
            if file_ccf[n][-27:] in filenames:
                with fits.open(file_ccf[n]) as hdulist:
                    header 	= hdulist[12].header
                    bjd[n] 	= header['CCFJDMOD']
                    rv[n] 	= header['CCFRVMOD']*1000
                    σrv[n]	= header['DVRMSMOD']*1000
                    ccf_per_order   = hdulist[12].data                
                    ccf_per_obs     = np.sum(ccf_per_order, axis=0)
                    v_grid 	= header['CCFSTART'] + np.arange(len(ccf_per_obs))*header['CCFSTEP']    # header['CCFSTART'] = -100; header['CCFSTEP'] = 0.25
                    if not np.any(CCF):
                        CCF = ccf_per_obs
                    else:
                        CCF = np.vstack((CCF, ccf_per_obs)) 
                np.savetxt(path + file_ccf[n][-27:-4] + 'ccf', ccf_per_order)
            bar()

    np.savetxt(path_prefix + single_date.strftime("ccf_by_day/%Y-%m-%d.CCF"), CCF)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))