import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import stats
import copy
import sys
sys.path.append("../")
from FIESTA_functions import *
import pandas as pd
import shutil


#--------------------------------------------------------------------
# import data 
#--------------------------------------------------------------------
# file_ccf	= sorted(glob.glob('../../AstroData/NEID_solar/*.fits'))
file_ccf	= sorted(glob.glob('../../AstroData/NEID_solar_0430_L2/*.fits'))
quality_df  	= pd.read_csv('20210430_solar_info.csv')
for n in range(N_file):
	if not (file_ccf[n][-20:] == quality_df['filename'][n][-20:]):
		print('File not matched!')

N_file		= len(file_ccf)
bjd 		= np.zeros(N_file)
rv 			= np.zeros(N_file)
σrv			= np.zeros(N_file)

CCF 		= []
for n in range(N_file):
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

# Select elegible observations 
'''
solar_hour_angle_threshold_good = 1.5
pyrhelio_ratio_min_good = 0.90
pyrhelio_ratio_max_good = 1.05
pyrhelio_rms_min_good = 0.0
pyrhelio_rms_max_good = 0.003
min_obs_binned_in_day = 20
'''
idx= (abs(quality_df['solar_hour_angle'])<1.5) & (quality_df['pyrheliometer_rms_flux']<0.003)
bjd 	= bjd[idx]
rv  	= rv[idx]
σrv 	= σrv[idx]
CCF 	= CCF[idx,:].T 	#(N_v, N_file)
σCCF 	= CCF**0.5
N_file 	= len(bjd) 	# update N_file
	
# plt.plot(hdulist[12].data.T)
# plt.show()


plt.rcParams.update({'font.size': 14})
alpha=0.5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

fig, axes = plt.subplots(figsize=(15, 3))
plt.gcf().subplots_adjust(bottom=0.2)
plt.errorbar((bjd-bjd[0])*24*60, rv-np.mean(rv), σrv, marker='.', color='black', ls='none', alpha=alpha)
# plt.errorbar((bjd[~idx]-bjd[0])*24*60, rv[~idx]-np.mean(rv), σrv[~idx], marker='.', color='red', ls='none', alpha=alpha, label='σ > 1 m/s')
plt.plot((bjd-bjd[0])*24*60, rv-np.mean(rv), 'k-', alpha=0.3, lw=2)
plt.title('NEID solar RV')
plt.xlabel('T [min]')
plt.ylabel('RV [m/s]')
plt.legend()
plt.savefig('NEID_solar_RV.png')
plt.show()





CCF_backup 	= CCF
σCCF_backup = σCCF

# plt.plot(v_grid, CCF)
# plt.show()

# CCF = CCF_backup
# σCCF = σCCF_backup

# Information of the hdulist
'''
	hdulist.info()
'''


'''
for n in range(N_file):
	hdulist = fits.open(file_ccf[n])
	bjd[n] 	= hdulist[12].header['CCFJDMOD']
	rv[n] 	= hdulist[12].header['CCFRVMOD']
	σrv[n]	= hdulist[12].header['DVRMSMOD']
'''
#--------------------------------------------------------------------
# data cleaning 
#--------------------------------------------------------------------
idx_nor = ((120<v_grid) & (v_grid<150)) | ((50<v_grid) & (v_grid<80))
idx_ccf = (80<v_grid) & (v_grid<120)

for n in range(N_file):
	σCCF[:,n] 	= σCCF[:,n] / np.mean(CCF[idx_nor,n])
	CCF[:,n] 	= CCF[:,n] / np.mean(CCF[idx_nor,n])
	

v_grid	= v_grid[idx_ccf]
CCF 	= CCF[idx_ccf,:]
σCCF 	= σCCF[idx_ccf,:]

idx_ccf = (v_grid>=84.50) & (v_grid<=114.25)
plt.plot(v_grid[idx_ccf], CCF[idx_ccf,:])
plt.title('CCF (overplotted)')
plt.xlabel('V grid [km/s]')
plt.ylabel('Normalised flux')
plt.savefig('CCF.png')
plt.show()

CCF 	= 1 - CCF
df, shift_spectrum, err_shift_spectrum, power_spectrum, err_power_spectrum, RV_gauss = FIESTA(v_grid, CCF, σCCF, k_max=5)
shift_spectrum 		*= 1000
err_shift_spectrum 	*= 1000
RV_gauss 			*= 1000

shift_function = np.zeros(shift_spectrum.shape)
for i in range(shift_spectrum.shape[0]):
	shift_function[i,:] = shift_spectrum[i,:] - (rv-np.mean(rv))

print(df.head(50).round({	'ξ'					: 3, 
					'individual_SNR'	: 1,
					'ts_SNR_A'			: 1, 
					'ts_SNR_ϕ'			: 1,
					'modelling noise'	: 5,
					'recoverable_CCF_SNR': 0
					}))

#--------------------------------------------------------------------
# plotting 
#--------------------------------------------------------------------
plt.rcParams.update({'font.size': 12})

def time_series(x, y, dy, N=None,
				ylabel='k=',
				title='Time series',
				file_name='Time_series.png'):
	if N==None:
		N = y.shape[1]
	plt.subplots(figsize=(12, N*0.8))

	for i in range(N):
		ax = plt.subplot(N, 1, i+1)
		if i == 0:
			plt.title(title)
		plt.errorbar(x, y[:, i], dy[:, i], marker='.', color='black', ls='none', alpha=0.5, ms=5)
		plt.ylabel(ylabel+str(i+1))
		if i != N-1:
			ax.set_xticks([])
		else:
			plt.xlabel('T [min]')
	plt.savefig(file_name)
	plt.show()

time_series(x=(bjd-bjd[0])*24*60, y=shift_spectrum[0:9,:].T, dy=err_shift_spectrum[30:40,:].T, N=None,
				title='$\Delta RV_k$ time series',
				file_name='FIESTA_shift_time_series.png')

time_series(x=(bjd-bjd[0])*24*60, y=power_spectrum[0:9,:].T, dy=err_power_spectrum[0:10,:].T, N=None,
				title='$A_k$ time series',
				file_name='FIESTA_A_time_series.png')

#--------------------------------------------------------------------
# visualize data 
#--------------------------------------------------------------------
plt.rcParams.update({'font.size': 14})
alpha=0.5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

fig, axes = plt.subplots(figsize=(15, 3))
plt.gcf().subplots_adjust(bottom=0.2)
plt.errorbar((bjd-bjd[0])*24*60, rv-np.mean(rv), σrv, marker='.', color='black', ls='none', alpha=alpha)
plt.title('NEID solar RV (5 Oct 2021)')
plt.xlabel('T [min]')
plt.ylabel('RV [m/s]')
plt.savefig('NEID_solar_RV.png')
plt.show()

#----------------------------
# Periodogram 
#----------------------------
from astropy.timeseries import LombScargle
frequency, power = LombScargle((bjd-bjd[0])*24*60, rv, σrv).autopower()
plt.plot(frequency, power)
plt.xlabel('$f$ [1/min]')
plt.ylim(0, 1.1*max(power))
plt.show()

plt.plot(1/frequency, power)
plt.xlim([1,(max(bjd)-min(bjd))*24*60/2])
plt.xscale('log')
plt.ylim(0, 1.1*max(power))
plt.xlabel('Period [min]')
plt.show()

#----------------------------
# correlations
#----------------------------
# for i in range(9):
# 	plt.plot(rv-np.mean(rv), shift_spectrum[i,:] - (rv-np.mean(rv)), '.')
# 	plt.xlabel('RV [m/s]')
# 	plt.ylabel('RV_k [m/s]')
# 	plt.grid()
# 	plt.show()

fig, ax = plt.subplots(3, 3)
fig.set_figheight(9)
fig.set_figwidth(9)
for i in range(9):
	plt.gcf().subplots_adjust(wspace=0.5, hspace=0.12)
	ax[int(i/3), i%3].plot(rv-np.mean(rv), power_spectrum[i,:], '.')
	# ax[int(i/3), i%3].plot(rv-np.mean(rv), shift_spectrum[i,:] - RV_gauss, '.')
	# ax[int(i/3), i%3].plot(rv-np.mean(rv), shift_spectrum[i,:] - (rv-np.mean(rv)), '.')
	ax[int(i/3), i%3].set_xlabel(r'$RV_{NEID}$ [m/s]')
	ax[int(i/3), i%3].set_ylabel(r'$A_%d$' %(i+1))
	ax[int(i/3), i%3].grid()
plt.show()


#----------------------------
# Time-series
#----------------------------
t_minute = (bjd-bjd[0])*24*60
from sklearn.linear_model import LinearRegression
k_mode 	= 5
alpha1, alpha2 = [0.5,0.2]
widths 	= [8,1]
heights = [1,1,1,1,1,1]
gs_kw 	= dict(width_ratios=widths, height_ratios=heights)
plt.rcParams.update({'font.size': 12})
fig6, f6_axes = plt.subplots(figsize=(10, k_mode+1), ncols=2, nrows=k_mode+1, constrained_layout=True,
                             gridspec_kw=gs_kw)
for r, row in enumerate(f6_axes):
	for c, ax in enumerate(row):		
		if c==0:
			if r==0:
				ax.errorbar(t_minute, rv-np.mean(rv), σrv, marker='.', color='black', ls='none', alpha=alpha1)
				ax.set_title('Time-series')
				ax.set_ylabel('$RV_{NEID}$')
			else:				
				ax.errorbar(t_minute, shift_function[r-1,:], err_shift_spectrum[r-1,:],  marker='.', color='black', ls='none', alpha=alpha1)
				ax.set_ylabel(r'$\Delta$RV$_{%d}$' %(r))
			if r!=k_mode:
				ax.set_xticks([])
			else:
				ax.set_xlabel('Time [min]')
		if c==1:
			if r==0:
				reg = LinearRegression().fit(rv.reshape(-1, 1), rv.reshape(-1, 1))
				score = reg.score(rv.reshape(-1, 1), rv.reshape(-1, 1))
				ax.set_title('score = {:.2f}'.format(score))
				ax.plot(rv-np.mean(rv), rv-np.mean(rv), 'k.', alpha = alpha2)				
			if r>0:
				reg = LinearRegression().fit(rv.reshape(-1, 1), shift_function[r-1,:].reshape(-1, 1))
				score = reg.score(rv.reshape(-1, 1), shift_function[r-1,:].reshape(-1, 1))
				ax.set_title('score = {:.2f}'.format(score))
				ax.plot(rv-np.mean(rv), shift_function[r-1,:], 'k.', alpha = alpha2)
			if r!=k_mode:
				ax.set_xticks([])
			else:
				ax.set_xlabel('$RV_{NEID}$')
			ax.yaxis.tick_right()
plt.savefig('time-series_and_shift_correlation.png')			
plt.show()


fig6, f6_axes = plt.subplots(figsize=(10, k_mode+1), ncols=2, nrows=k_mode+1, constrained_layout=True,
                             gridspec_kw=gs_kw)
for r, row in enumerate(f6_axes):
	for c, ax in enumerate(row):		
		if c==0:
			if r==0:
				ax.errorbar(t_minute, rv-np.mean(rv), σrv, marker='.', color='black', ls='none', alpha=alpha1)
				ax.set_title('Time-series')
				ax.set_ylabel('$RV_{NEID}$')
			else:				
				ax.errorbar(t_minute, power_spectrum[r-1,:], err_power_spectrum[r-1,:],  marker='.', color='black', ls='none', alpha=alpha1)
				ax.set_ylabel(r'$A_{%d}$' %(r))
			if r!=k_mode:
				ax.set_xticks([])
			else:
				ax.set_xlabel('Time [min]')
		if c==1:
			if r==0:
				reg = LinearRegression().fit(rv.reshape(-1, 1), rv.reshape(-1, 1))
				score = reg.score(rv.reshape(-1, 1), rv.reshape(-1, 1))
				ax.set_title('score = {:.2f}'.format(score))
				ax.plot(rv-np.mean(rv), rv-np.mean(rv), 'k.', alpha = alpha2)				
			if r>0:
				reg = LinearRegression().fit(rv.reshape(-1, 1), power_spectrum[r-1,:].reshape(-1, 1))
				score = reg.score(rv.reshape(-1, 1), power_spectrum[r-1,:].reshape(-1, 1))
				ax.set_title('score = {:.2f}'.format(score))
				ax.plot(rv-np.mean(rv), power_spectrum[r-1,:], 'k.', alpha = alpha2)
			if r!=k_mode:
				ax.set_xticks([])
			else:
				ax.set_xlabel('$RV_{NEID}$')
			ax.yaxis.tick_right()
plt.savefig('time-series_and_A_correlation.png')
plt.show()



#----------------------------
# Periodogram 
#----------------------------

def periodogram(x, y, dy, N=None,
				plot_min_t=1, study_min_t=5, max_f=1, spp=100, xc=None,
				ylabel=None,
				title = 'Periodogram',
				file_name='Periodogram.png'):
	
	from scipy.signal import find_peaks

	if N==None:
		N = y.shape[1]
	time_span = (max(x) - min(x))
	min_f   = 1/time_span

	plt.subplots(figsize=(10, N+1))

	for i in range(N):
		ax = plt.subplot(N,1,i+1)
		if i == 0:
			plt.title(title)

		frequency, power = LombScargle(x, y[:, i], dy[:, i]).autopower(minimum_frequency=min_f,
													   maximum_frequency=max_f,
													   samples_per_peak=spp)

		plot_x = 1/frequency
		idxx = (plot_x>plot_min_t) & (plot_x<100)
		height = max(power[idxx])*0.75
		plt.plot(plot_x[idxx], power[idxx], 'k-', label=r'$\xi$'+str(i+1), alpha=0.5)
		peaks, _ = find_peaks(power[idxx], height=height)
		plt.plot(plot_x[idxx][peaks], power[idxx][peaks], "ro")
		if xc != None:
			plt.axvline(x=xc, color='k', linestyle='--', alpha = 0.5)

		for n in range(len(plot_x[idxx][peaks])):
			plt.text(plot_x[idxx][peaks][n], power[idxx][peaks][n], '%.1f' % plot_x[idxx][peaks][n], fontsize=10)

		plt.xlim([plot_min_t,100])
		plt.ylim([0,1.5*height])
		plt.xscale('log')
		if i==0:
			plt.ylabel('NEID')
		if i>0:
			plt.ylabel(ylabel+str(i))

		if i != N-1:
			ax.set_xticks([])
		else:
			plt.xlabel('Period [min]')

	plt.savefig(file_name)
	plt.show()


periodogram(x=t_minute, y=np.vstack((rv, power_spectrum)).T, dy=np.vstack((σrv, err_power_spectrum)).T, N=6,
			plot_min_t=1.3, study_min_t=1, max_f=1, spp=100,
			ylabel='k=',
			file_name='FIESTA_amplitude_periodogram.png')

periodogram(x=t_minute, y=np.vstack((rv, shift_spectrum)).T, dy=np.vstack((σrv, err_shift_spectrum)).T, N=6,
			plot_min_t=1.3, study_min_t=1, max_f=1, spp=100,
			ylabel='k=',
			file_name='FIESTA_shift_periodogram.png')
