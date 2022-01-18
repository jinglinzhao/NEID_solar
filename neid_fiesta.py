import numpy as np
import os
import glob
import pandas as pd
from astropy.io import fits
from datetime import timedelta, date
from datetime import datetime
from alive_progress import alive_bar

quality_df  = pd.read_csv('combined_rvs_1.csv')
filenames 	= [quality_df['Filename'][i][-27:] for i in range(len(quality_df))]

start_date  = date(2020, 6, 1)
end_date    = date(2020, 6, 30)

start_time  = datetime.now()
for single_date in daterange(start_date, end_date):

    print(single_date.strftime("%Y-%m-%d"))
    
