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
quality_df  = pd.read_csv('./lib/combined_rvs_1.csv')
filenames   = [quality_df['Filename'][i][-27:] for i in range(len(quality_df))]
path_prefix = '/storage/group/ebf11/default/jvz5625/'

start_date  = date(2021, 1, 1)
end_date  = date(2021, 1, 2)
# end_date    = date(2021, 12, 31)

plot        = False
o_start     = 55    # 56 in Julia (physical 118)
o_end       = 108   # 108 in julia (physical 65)
o_exclude   = 173 - np.array([66, 67, 68, 69, 73, 74, 75, 76, 78, 83, 107,108,112,113,117, 118]) # -> version 4 (data_v4);
o_used      = np.array([x for x in np.arange(o_start, o_end) if (x in o_exclude) == False])

v_grid      = -100 + np.arange(1604)*0.25
idx_v       = (v_grid>87) & (v_grid<111)

CCF, σCCF                       = [], []
bjd, rv, σrv                    = np.array([]), np.array([]), np.array([])
CCF_daily, σCCF_daily           = [], []
bjd_daily, rv_daily, σrv_daily  = np.array([]), np.array([]), np.array([])

CCF_used_orders = []

start_time  = datetime.now()
for single_date in daterange(start_date, end_date):

    print(single_date.strftime("%Y-%m-%d"))

    path        = path_prefix + single_date.strftime('extracted_ccf/%m/%d/')
    file_ccf    = sorted(glob.glob(path + '/*.ccf')) 
    N_file      = len(file_ccf)

    if N_file != 0:
        with alive_bar(N_file) as bar:
            for n in range(N_file):
                ccf_per_order   = np.loadtxt(file_ccf[n])
                CCF_used_orders.append(ccf_per_order[o_used, :])
                bar()

CCF_used_orders = np.stack(CCF_used_orders)                        
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# np.save('./lib/CCF_used_orders.npy', CCF_used_orders)
# CCF_used_orders = np.load('./lib/CCF_used_orders.npy')