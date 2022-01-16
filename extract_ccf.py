import glob
from astropy.io import fits
import pandas as pd
from datetime import timedelta, date

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
filenames = [quality_df['Filename'][i][-27:] for i in range(len(quality_df))]

start_date = date(2020, 6, 1)
end_date = date(2020, 6, 1)
for date in daterange(start_date, end_date):
    file_ccf = sorted(glob.glob('/gpfs/group/ebf11/default/pipeline/data/neid_solar/v1.1/L2/2021/' + \
                                date.strftime('%m') +'/' + date.strftime('%d') + '/*.fits')) 

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
                ccf  	= np.sum(hdulist[12].data.T, axis=1)
                v_grid 	= header['CCFSTART'] + np.arange(len(ccf))*header['CCFSTEP']
                if n == 0:
                    CCF = ccf
                else:
                    CCF = np.vstack((CCF, ccf))    