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
rssp=1.700
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
            b = b + P(m, l, np.cos(t)) * (g(m, l, B_r) * np.cos(m * p) + h(m, l, B_r) * np.sin(m * p)) * z
    return b


def compare_groups(numbers):
    positives = [x for x in numbers if x > 0]
    negatives = [x for x in numbers if x < 0]
    top_positives = sorted(positives, reverse=True)[:15]
    top_negatives = sorted(negatives, key=lambda x: abs(x), reverse=True)[:15]
    sum_pos = sum(top_positives)
    sum_neg_abs = sum(abs(x) for x in top_negatives)
    if sum_pos >= sum_neg_abs:
        return top_positives, sum_pos
    else:
        return top_negatives,sum_neg_abs
dt=60
def Tenminute(a,time):

    minutes = np.array([t.timestamp() // (60*dt) for t in time])
    unique_minutes = np.unique(minutes)
    average_per_minute = np.array([np.mean(a[minutes == m]) for m in unique_minutes])
    vectorized_fromtimestamp = np.vectorize(datetime.fromtimestamp)
    datetime_objects = vectorized_fromtimestamp(unique_minutes*(60*dt))
    return average_per_minute,datetime_objects

ss=np.array([1.312,1.241, 1.281,1.91372, 1.188,1.8583, 1.304828, 2.116964, 1.4826, 1.591, 1.6715, 2.40, 1.2923, 2.1556,1.50749, 2.3151, 1.7797, 1.66679, 2.0053])
print(ss)
sequences = []
for right in ss:
    seq = np.linspace(start=rssp, stop=right, num=5)
    sequences.append(seq.round(3).tolist())
print(sequences)
sequence = [f"{i:02d}" for i in range(1, 20)]
fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(5, 4, hspace=0.365, wspace=0.275) 
for i in range(5):
    for j in range(4):

        if i == 4 and j == 3:
            break
        else:
            k=4*i+j
            #GONG and coordination
            hdul_mag = fits.open(filePath_GONG + ID_GONG[k] + '/' + ID_GONG[k])
            gong_map = Map(filePath_GONG + ID_GONG[k] + '/' + ID_GONG[k])
            lat = np.radians(sunpy.map.all_coordinates_from_map(gong_map).lat)
            lon = np.radians(sunpy.map.all_coordinates_from_map(gong_map).lon)
            lat = u.Quantity(lat)
            lon = u.Quantity(lon)
            angle_lat = np.pi / 2 - lat.to(u.rad).value
            angle_lon = lon.to(u.rad).value
            X = angle_lon
            Y = angle_lat[::-1]
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

            ax = plt.subplot(gs[i, j])
            ax.scatter(times[Brpsp > 0], Brpsp[Brpsp > 0], color='#E64A19', s=0.35, label='Positive $B_r$')
            ax.scatter(times[Brpsp < 0], Brpsp[Brpsp < 0], color='#1976D2', s=0.35, label='Negative $B_r$')
            phiss = spiral(phipsp, rpsp, 2.5, vresult)
            Brs = Bss(2.5, thetapsp, phiss, B_r, L) * 2.5 * 2.5 / (rpsp ** 2) * 1e+5
            Brs = np.interp(np.arange(len(Brs)), np.flatnonzero(~np.isnan(Brs)), Brs[~np.isnan(Brs)])
            ax.scatter(times, Brs,
                               s=0.35,
                               c='black',
                               linewidths=0.2,
                               alpha=0.85,
                               label=f'$2.5R_s$'
                               )
            colors = ['#D2BF9F', '#EB9505', '#5D74A2', '#EFCF36', '#A4799E']
            for l in range(len(sequences[k])):
                phiss2 = spiral(phipsp, rpsp, sequences[k][l], vresult)
                mag = Bss(sequences[k][l], thetapsp, phiss2, B_r, L) * sequences[k][l] * sequences[k][l] / (rpsp ** 2) * 1e+5
                mag = np.interp(np.arange(len(mag)), np.flatnonzero(~np.isnan(mag)), mag[~np.isnan(mag)])
                ax.scatter(times, mag, color=colors[l], s=0.3, label='PFSS Simulation'+str(l))


            #ax.set_xlabel('Date', fontsize=5)
            ax.set_ylabel('$B_r$ / nT', fontsize=7)
            if j>=1:
                ax.set_ylabel('')
            ax.set_title(f'PSP Encounter' + sequence[k], fontsize=8)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
            def custom_format(value, tick_number):
                return f"{value:.1f}"
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f"{value:.1f}"))
            date_format = mdates.DateFormatter('%y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.tick_params(axis='x', which='major', labelsize=7)
            ax.tick_params(axis='y', which='major', labelsize=7)



plt.savefig('Simulations2.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()