
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from datetime import datetime

from FIESTA_functions import *
from HARPS_N_functions import *

#----------------------------------
# Read data
#----------------------------------
if 0:
    v_grid  = np.loadtxt('./data_526_625/v_grid.txt')
    CCF     = np.loadtxt('./data_526_625/CCF.txt')
    σ_CCF   = np.loadtxt('./data_526_625/σ_CCF.txt')
    bjd     = np.loadtxt('./data_526_625/bjd.txt')
    rv      = np.loadtxt('./data_526_625/rv.txt')
    σrv     = np.loadtxt('./data_526_625/σrv.txt')

if 1:
    v_grid  = np.loadtxt('./data/v_grid.txt')
    CCF     = np.loadtxt('./data/CCF.txt')
    σ_CCF   = np.loadtxt('./data/σ_CCF.txt')
    bjd     = np.loadtxt('./data/bjd.txt')
    rv      = np.loadtxt('./data/rv.txt')
    σrv     = np.loadtxt('./data/σrv.txt')

plt.plot(v_grid, CCF)
plt.show()

#==============================================================================
# Feed CCFs into FIESTA
#==============================================================================
df, shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(v_grid, CCF, σ_CCF, k_max=6)
shift_spectrum      *= 1000
err_shift_spectrum  *= 1000
RV_gauss            *= 1000
shift_function      = np.zeros(shift_spectrum.shape)

for i in range(shift_spectrum.shape[0]):
    shift_function[i,:] = shift_spectrum[i,:] - rv # look back; change rv_raw_daily

#----------------------------------
# Plot the RVs 
#----------------------------------

def plot_rv(date1, date2):
    plt.rcParams.update({'font.size': 14})
    alpha   = 0.3
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    idx_bjd = (bjd>date1) & (bjd<date2+1)

    fig, axes = plt.subplots(figsize=(15, 3))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.errorbar(bjd[idx_bjd]-2400000, rv[idx_bjd]-np.mean(rv[idx_bjd]), σrv[idx_bjd], c='red', marker='.', ls='none', alpha=alpha, label='rv')
    plt.errorbar(bjd[idx_bjd]-2400000, RV_gauss[idx_bjd]-np.mean(RV_gauss[idx_bjd]), σrv[idx_bjd], c='blue', marker='.', ls='none', alpha=alpha, label='RV_gauss')
    plt.legend()
    plt.xlabel('BJD - 2400000 [d]')
    plt.ylabel('RV [m/s]')

    if date1 == date2:
        filename = '%d.%d.%d' %(tuple(pyasl.daycnv(date1))[:3])
        plt.title('%d.%d.%d' %tuple(pyasl.daycnv(date1))[:3])
    else:
        filename = '%d.%d.%d-%d.%d.%d' %(tuple(pyasl.daycnv(date1))[:3]+tuple(pyasl.daycnv(date2))[:3])
        plt.title(filename)
    plt.savefig('./figure/' + filename+'.png')
    plt.show()

T1  = pyasl.jdcnv(datetime(2021, 5, 26))
T2  = pyasl.jdcnv(datetime(2021, 6, 25))
plot_rv(T1, T2)

T1  = pyasl.jdcnv(datetime(2021, 1, 1))
T2  = pyasl.jdcnv(datetime(2021, 12, 31))
plot_rv(T1, T2)

for i in range(int(T2-T1)):
    T = pyasl.jdcnv(datetime(2021, 5, 26)) + i 
    plot_rv(T, T)

#----------------------------------
# FIESTA
#----------------------------------
T1      = pyasl.jdcnv(datetime(2021, 5, 26))
T2      = pyasl.jdcnv(datetime(2021, 6, 25))
idx_bjd = (bjd>T1) & (bjd<T2+1)

plt.rcParams.update({'font.size': 14})
plot_all(k_mode=6, t=bjd[idx_bjd], rv=rv[idx_bjd], erv=σrv[idx_bjd], 
    ind=power_spectrum[:,idx_bjd], eind=err_power_spectrum[:,idx_bjd], 
    ts_xlabel='BJD - 2400000 [d]', 
    rv_xlabel='$RV_{NEID}$', 
    pe_xlabel='Period [days]',
    ind_yalbel=r'$A$',
    file_name='./figure/' + 'Amplitude_time-series_correlation_periodogram_NEID.pdf')

plot_all(k_mode=6, t=bjd[idx_bjd], rv=rv[idx_bjd], erv=σrv[idx_bjd],  
    ind=shift_function[:,idx_bjd], eind=err_shift_spectrum[:,idx_bjd], 
    ts_xlabel='BJD - 2400000 [d]', 
    rv_xlabel='$RV_{HARPS}$', 
    pe_xlabel='Period [days]',
    ind_yalbel=r'$\Delta RV$',
    file_name='./figure/' + 'shift_time-series_correlation_periodogram_SCALPELS.pdf')
plt.show()


import pandas as pd
quality_df  = pd.read_csv('combined_rvs_1.csv')
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(figsize=(15, 3))
alpha=0.3
plt.plot(bjd-2400000, quality_df['CaIIHK'], '.', label='CaIIHK',alpha=alpha)
plt.plot(bjd-2400000, quality_df['Ha06_1'], '.', label='Ha06_1',alpha=alpha)
plt.plot(bjd-2400000, quality_df['Ha06_2'], '.', label='Ha06_2',alpha=alpha)
plt.plot(bjd-2400000, quality_df['Ha06_3'], '.', label='Ha06_3',alpha=alpha)
xlabel='BJD - 2400000 [d]'
plt.legend()
plt.show()

#==============================================================================
# Testing
#==============================================================================
if 0:

    # pyasl.jdcnv(datetime(2020, 1, 1))

    time_in_day = bjd - np.array([int(bjd[i]) for i in range(len(bjd))])
    plt.hist(time_in_day)
    plt.show()

if 0:
    dates = ["01/02/2020", "01/03/2020", "01/04/2020"]
    x_values = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in dates]
    y_values = [1, 2, 3]

    ax = plt.gca()

    formatter = mdates.DateFormatter("%Y-%m-%d")

    ax.xaxis.set_major_formatter(formatter)

    locator = mdates.DayLocator()

    ax.xaxis.set_major_locator(locator)

    plt.plot(x_values, y_values)