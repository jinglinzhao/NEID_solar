#----------------------------------
# Read data
#----------------------------------
v_grid  = np.loadtxt('./data/v_grid.txt')
CCF     = np.loadtxt('./data/CCF.txt')
σ_CCF   = np.loadtxt('./data/σ_CCF.txt')

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

    idx_bjd = (bjd>date1) & (bjd<date2)

    fig, axes = plt.subplots(figsize=(15, 3))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.errorbar(bjd[idx_bjd]-2400000, rv[idx_bjd]-np.mean(rv[idx_bjd]), σrv, c='purple', marker='.', ls='none', alpha=alpha, label='rv')
    plt.errorbar(bjd[idx_bjd]-2400000, RV_gauss[idx_bjd]-np.mean(RV_gauss[idx_bjd]), σrv, c='black', marker='.', ls='none', alpha=alpha, label='RV_gauss')
    plt.legend()
    plt.xlabel('BJD - 2400000 [d]')
    plt.ylabel('RV [m/s]')
    # plt.savefig('rv_daily.pdf')
    plt.show()


from PyAstronomy import pyasl
pyasl.daycnv(bjd[0])

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