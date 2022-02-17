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

start_date  = date(2021, 1, 1)
end_date    = date(2021, 12, 31)

start_time  = datetime.now()
for single_date in daterange(start_date, end_date):

    print(single_date.strftime("%Y-%m-%d"))

    path = path_prefix + 'extracted_ccf/' + single_date.strftime('%m') + '/' + single_date.strftime('%d') + '/'

    file_ccf = sorted(glob.glob('/gpfs/group/ebf11/default/pipeline/data/neid_solar/v1.1/L2/2021/' + \
                                single_date.strftime('%m') + '/' + single_date.strftime('%d') + '/*.fits')) 
    N_file		= len(file_ccf)
    
    with alive_bar(N_file) as bar:
        for n in range(N_file):
            if file_ccf[n][-27:] in filenames:
                with fits.open(file_ccf[n]) as hdulist:
                    header 	= hdulist[12].header
                    
                    for index in range(173-52+1):
                        order_number = 173-index
                        if order_number < 100:
                            order_name = 'CCFRV0' + str(order_number)
                        else:
                            order_name = 'CCFRV' + str(order_number)
                        quality_df.loc[quality_df['Filename']==file_ccf[n], order_name] = header[order_name]
            bar()     

quality_df.to_csv('full_combined_rvs_1.csv')    

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))           