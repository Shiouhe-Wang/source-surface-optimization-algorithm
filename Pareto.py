#basis
import numpy as np
import os
os.environ["CDF_LIB"] = "D:/cdf3.8.0_64bit_VS2015/lib"
from scipy import special
from astropy.io import fits
from scipy.special import factorial
from datetime import datetime, timedelta

#astropy
import sunpy.map
from sunpy.map import Map
import astropy.constants as const
import astropy.units as u
import sunpy.map
from astropy.coordinates import SkyCoord
from pfsspy.sample_data import get_gong_map
from spacepy import pycdf

#PLOT
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
import sys
sys.path.insert(0, '../../Utilities/')

#The Sun
rsun=6.963e+8
rssp=2.000
rss=rssp*rsun
rsunp = 6.963e+5

#mesh
N=180
start = 0.5*np.pi/180
stop = 179.5*np.pi/180
theta = np.linspace(start, stop, num=180)
phi = np.array([i * np.pi / N for i in range(360)])
n=len(theta)
X, Y = np.meshgrid(phi, theta)

#GONG
filePath_GONG = 'D:/Nonsphere/PAPER3/Encounter/GONG/'
ID_GONG=os.listdir(filePath_GONG)

#Orbit
filePath_orbit = 'D:/Nonsphere/PAPER3/Encounter/Orbit/'
ID_orbit=os.listdir(filePath_orbit)

#magnetic
filePath_mag = 'D:/Nonsphere/PAPER3/Encounter/Fields/'
ID_mag=os.listdir(filePath_mag)

vsw = 360
w = 2*np.pi/(24.47*24*3600)
def spiral(phipsp,rpsp,r,vsw):
    phir=phipsp-w*rsunp*(r-rpsp)/vsw
    return phir
dtype = [
    ('Year', 'i4'), ('Month', 'i4'), ('Day', 'i4'),
    ('Hour', 'i4'), ('Minute', 'i4'), ('Second', 'i4'),
    ('HG_X', 'f8'), ('HG_Y', 'f8'), ('HG_Z', 'f8')
]
dt=40
def Tenminute(a,time):

    minutes = np.array([t.timestamp() // (60*dt) for t in time])
    unique_minutes = np.unique(minutes)
    average_per_minute = np.array([np.mean(a[minutes == m]) for m in unique_minutes])
    vectorized_fromtimestamp = np.vectorize(datetime.fromtimestamp)
    datetime_objects = vectorized_fromtimestamp(unique_minutes*(60*dt))
    return average_per_minute,datetime_objects
def P(m,l,x):
    if m==0:
        p=(-1)**m*np.sqrt(factorial(l-m)/factorial(l+m))*special.lpmv(m, l, x)
    else:
        p=(-1)**m*np.sqrt(2*factorial(l-m)/factorial(l+m))*special.lpmv(m, l, x)
    return p
def g(m,l,B_r):
    f=np.cos(m * X)*P(m,l,np.cos(Y))*B_r
    return (2*l+1)*np.sum(f)/(180*360)
def h(m,l,B_r):
    f = np.sin(m * X)*P(m,l,np.cos(Y))*B_r
    return (2*l+1)*np.sum(f)/(180*360)
def Br(r,t,p,B_r,L):
    b=0
    for l in range(L):
        z=(rsun/r)**(l+2)*(l + 1 + l * (r / rss) ** (2 * l + 1))/(l + 1 + l * (rsun / rss) ** (2 * l + 1))
        for m in range(l+1):
            b=b+P(m, l, np.cos(t))*(g(m,l,B_r)*np.cos(m*p)+h(m,l,B_r)*np.sin(m*p))*z
    return b
def Bss(rssm,t,p,B_r,L):
    b = 0
    for l in range(1,L+1):
        z = (1 / rssm) ** (l + 2) * ((2 * l + 1) / (l + 1 + l * (1 / rssm) ** (2 * l + 1)))
        for m in range(l + 1):
            print(l, m)
            b = b + P(m, l, np.cos(t)) * (g(m, l, B_r) * np.cos(m * p) + h(m, l, B_r) * np.sin(m * p)) * z
            print(g(m, l, B_r),h(m, l, B_r))
    return b

np.random.seed(0)
sequence = ["02","03","05","07"]

k1=np.array([2,3,5,7])
rssop=np.array([1.241, 1.281,1.188,1.304828])
rsspo=np.array([2.05, 1.65,1.85,2.15])
fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.2)  # 设置hspace为默认值，wspace为0
for i in range(2):
    for j in range(2):

        if i == 4 and j == 3:
            break
        else:
            u=2*i+j
            k=k1[u]-1
            #GONG and coordination
            hdul_mag = fits.open(filePath_GONG + ID_GONG[k] + '/' + ID_GONG[k])
            B_r = np.array(hdul_mag[0].data)[::-1]
            # psp
            directory = 'D:/Nonsphere/PAPER3/Encounter/Fields/' + ID_mag[k]
            cdf = pycdf.CDF(os.listdir(directory)[0])
            timeb = np.array(cdf["epoch_mag_RTN_1min"])
            B = np.array(cdf["psp_fld_l2_mag_RTN_1min"])
            cdf.close()
            for m in range(len(os.listdir(directory))):
                cdf = pycdf.CDF(os.listdir(directory)[m])
                timeb = np.append(timeb, cdf["epoch_mag_RTN_1min"])
                B = np.concatenate((B, cdf["psp_fld_l2_mag_RTN_1min"]), axis=0)
                cdf.close()
            Br = B[:, 0:1].reshape(1, -1)[0]
            times = timeb
            #IQR
            data_by_window = {}
            for t, v in zip(times, Br):
                window_start = t - timedelta(minutes=t.minute % 30)
                window_key = window_start.strftime('%Y-%m-%d %H:%M')
                if window_key not in data_by_window:
                    data_by_window[window_key] = {'times': [], 'values': []}
                data_by_window[window_key]['times'].append(t)
                data_by_window[window_key]['values'].append(v)
            cleaned_times = []
            cleaned_values = []
            outliers = []
            for window, data in data_by_window.items():
                values = np.array(data['values'])
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                for t, v in zip(data['times'], values):
                    if lower_bound <= v <= upper_bound:
                        cleaned_times.append(t)
                        cleaned_values.append(v)
                    else:
                        outliers.append((t, v))
            times = np.array(cleaned_times)
            Brpsp = np.array(cleaned_values)
            Brpsp, times = Tenminute(Brpsp, times)
            Brpsp = np.interp(np.arange(len(Brpsp)), np.flatnonzero(~np.isnan(Brpsp)), Brpsp[~np.isnan(Brpsp)])
            #PSP sweap
            directory = 'D:/Nonsphere/PAPER3/Encounter/Sweap/Encounter (' + str(k + 1) + ')'
            cdf = pycdf.CDF(os.listdir(directory)[0])
            timev = np.array(cdf["Epoch"])
            v = np.array(cdf["VEL_RTN_SUN"])
            cdf.close()
            for m in range(len(os.listdir(directory))):
                cdf = pycdf.CDF(os.listdir(directory)[m])
                timev = np.append(timev, cdf["Epoch"])
                v = np.concatenate((v, cdf["VEL_RTN_SUN"]), axis=0)
                cdf.close()
            vrpsp, timevr = Tenminute(v[:, 0:1], timev)
            vrpsp = np.interp(np.arange(len(vrpsp)), np.flatnonzero(~np.isnan(vrpsp)), vrpsp[~np.isnan(vrpsp)])
            positions = [np.where(times == value)[0] for value in timevr]
            positions = np.concatenate(positions).tolist()
            vresult = np.full(len(Brpsp), np.nan)
            positionss = [np.where(timevr == value)[0] for value in times[positions]]
            positionss = np.concatenate(positionss).tolist()
            vresult[positions] = vrpsp[positionss]
            vresult = np.interp(np.arange(len(vresult)), np.flatnonzero(~np.isnan(vresult)), vresult[~np.isnan(vresult)])

            data = np.genfromtxt('D:/Nonsphere/PAPER3/Encounter/Orbit/' + ID_orbit[k], dtype=dtype, skip_header=1)
            year, month, day, hour, minute = data['Year'], data['Month'], data['Day'], data['Hour'], data['Minute']
            datetime_array = np.array([datetime(year[k], month[k], day[k], hour[k], minute[k]) for k in range(len(year))])
            matching_indices = np.where(datetime_array == times[:, None])[1]
            data = data[matching_indices]
            x = data['HG_X'] / rsunp
            y = data['HG_Y'] / rsunp
            z = data['HG_Z'] / rsunp
            rpsp = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            thetapsp = np.arccos(z / rpsp)
            phipsp = np.arctan2(y, x)
            L = 22
            phiss2 = spiral(phipsp, rpsp, 2.5, vresult)
            Brs2 = Bss(2.5, thetapsp, phiss2, B_r, L) * 2.5 * 2.5 / (rpsp ** 2) * 1e+5
            Brs2 = np.interp(np.arange(len(Brs2)), np.flatnonzero(~np.isnan(Brs2)), Brs2[~np.isnan(Brs2)])
            c02 = np.max(np.abs(Brpsp)) / np.max(np.abs(Brs2))
            mag2 = c02 * Brs2

            phiss = spiral(phipsp, rpsp, rsspo[u], vresult)
            Brs = Bss(rsspo[u], thetapsp, phiss, B_r, L) * rsspo[u] * rsspo[u] / (rpsp ** 2) * 1e+5
            Brs = np.interp(np.arange(len(Brs)), np.flatnonzero(~np.isnan(Brs)), Brs[~np.isnan(Brs)])
            c01 = np.max(np.abs(Brpsp)) / np.max(np.abs(Brs))
            mag = c01 * Brs

            phissu = spiral(phipsp, rpsp, rssop[u], vresult)
            Brs = Bss(rssop[u], thetapsp, phissu, B_r, L) *  rssop[u] *  rssop[u] / (rpsp ** 2) * 1e+5
            Brs = np.interp(np.arange(len(Brs)), np.flatnonzero(~np.isnan(Brs)), Brs[~np.isnan(Brs)])
            c0 = np.max(np.abs(Brpsp)) / np.max(np.abs(Brs))
            magu = c0 * Brs
            ax = plt.subplot(gs[i, j])
            ax.scatter(times[Brpsp > 0], Brpsp[Brpsp > 0], color='#E64A19', s=2.2)
            ax.scatter(times[Brpsp < 0], Brpsp[Brpsp < 0], color='#1976D2', s=2.2)
            ax.plot(times, mag, '-', c='black', linewidth=2.3, label='$R_{ss}=2.50R_{s}$'+', scalar:'+str(c02)[0:5])
            ax.plot(times, mag2, '--', c='#D2BF9F', linewidth=2.3, label='$R_{ss}=$'+str(rsspo[u])+'$R_{s}$, scalar:'+str(c01)[0:5])
            ax.plot(times, magu, '--', c='#A4799E', linewidth=2.3, label='$R_{ss}=$'+str(rssop[u])[0:4]+'$R_{s}$, scalar:'+str(c0)[0:4])
            #ax.set_xlabel('Date', fontsize=5)
            ax.set_ylabel(r'$\bf{B_r}$ / nT', fontsize=10,fontweight='bold')
            if j>=1:
                ax.set_ylabel('')
            ax.set_title(f'PSP Encounter' + sequence[u], fontsize=10,fontweight='bold')
            ax.axhline(0, color='black', linewidth=1.0)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
            def custom_format(value, tick_number):
                return f"{value:.1f}"
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f"{value:.1f}"))
            date_format = mdates.DateFormatter('%y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.tick_params(axis='x', which='major', labelsize=8)
            ax.tick_params(axis='y', which='major', labelsize=10)
            ax.legend(fontsize=8,loc='lower left')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')

plt.savefig('Pareto.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

