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
dt=20
def Tenminute(a,time):

    minutes = np.array([t.timestamp() // (60*dt) for t in time])
    unique_minutes = np.unique(minutes)
    average_per_minute = np.array([np.mean(a[minutes == m]) for m in unique_minutes])
    vectorized_fromtimestamp = np.vectorize(datetime.fromtimestamp)
    datetime_objects = vectorized_fromtimestamp(unique_minutes*(60*dt))
    return average_per_minute,datetime_objects

sequence = [f"{i:02d}" for i in range(1, 20)]
for i in range(19):

    hdul_mag = fits.open(filePath_GONG + ID_GONG[i] + '/' + ID_GONG[i])
    gong_map = Map(filePath_GONG + ID_GONG[i] + '/' + ID_GONG[i])
    lat = np.radians(sunpy.map.all_coordinates_from_map(gong_map).lat)
    lon = np.radians(sunpy.map.all_coordinates_from_map(gong_map).lon)
    lat = u.Quantity(lat)
    lon = u.Quantity(lon)
    angle_lat = np.pi / 2 - lat.to(u.rad).value
    angle_lon = lon.to(u.rad).value
    X = angle_lon
    Y = angle_lat[::-1]
    B_r = np.array(hdul_mag[0].data)[::-1]

    directory = 'D:/Nonsphere/PAPER3/Encounter/Fields/' +  ID_mag[i]
    cdf = pycdf.CDF(os.listdir(directory)[0])
    timeb = np.array(cdf["epoch_mag_RTN_1min"])
    B = np.array(cdf["psp_fld_l2_mag_RTN_1min"])
    cdf.close()
    for j in range(len(os.listdir(directory))):
        cdf = pycdf.CDF(os.listdir(directory)[j])
        timeb = np.append(timeb, cdf["epoch_mag_RTN_1min"])
        B = np.concatenate((B, cdf["psp_fld_l2_mag_RTN_1min"]), axis=0)
        cdf.close()
    Br = B[:, 0:1].reshape(1, -1)[0]
    times = timeb
    data_by_window = {}

    for t, v in zip(times, Br):
        window_start = t - timedelta(minutes=t.minute % dt)
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
    directory = 'D:/Nonsphere/PAPER3/Encounter/Sweap/Encounter (' + str(i + 1) + ')'
    cdf = pycdf.CDF(os.listdir(directory)[0])
    timev = np.array(cdf["Epoch"])
    v = np.array(cdf["VEL_RTN_SUN"])
    cdf.close()
    for j in range(len(os.listdir(directory))):
        cdf = pycdf.CDF(os.listdir(directory)[j])
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

    data = np.genfromtxt('D:/Nonsphere/PAPER3/Encounter/Orbit/' + ID_orbit[i], dtype=dtype, skip_header=1)
    year, month, day, hour, minute=data['Year'], data['Month'], data['Day'], data['Hour'], data['Minute']
    datetime_array = np.array([datetime(year[k], month[k], day[k], hour[k], minute[k]) for k in range(len(year))])
    matching_indices = np.where(datetime_array == times[:, None])[1]
    data=data[matching_indices]
    x = data['HG_X'] / rsunp
    y = data['HG_Y'] / rsunp
    z = data['HG_Z'] / rsunp
    rpsp = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    thetapsp = np.arccos(z / rpsp)
    phipsp = np.arctan2(y, x)
    phiss = spiral(phipsp,rpsp,rssp,vresult)

    mt, ut = compare_groups(Brpsp)
    mean = np.mean(Brpsp, axis=0)
    std = np.std(Brpsp, axis=0, ddof=1)
    print(Brpsp)

    Brpsp = (Brpsp - mean) / std

    L = 22

    def J(rssp,thetapsp,phiss,B_r,L):
        Brs= Bss(rssp, thetapsp, phiss, B_r, L)*1e+5*rssp*rssp/(rpsp*rpsp)
        Brs=(Brs-mean)/std
        mse = np.sum((Brpsp-Brs) **2) / (2*len(Brs))
        print(mse)

        return mse

    ee = 1e-5
    def dJ(rssp, thetapsp, phiss, B_r, L):
        return (J(rssp + ee, thetapsp, phiss, B_r, L) - J(rssp - ee, thetapsp, phiss, B_r, L)) / (2 * ee)

    learning_rate = 0.3
    num_iters = 50

    def gradient_descent(rssp, thetapsp, phiss, B_r, L, learning_rate, num_iters):

        loss_history = np.zeros(num_iters)
        para = np.zeros(num_iters)
        grad = np.zeros(num_iters)

        for p in range(num_iters):

            gradient=dJ(rssp, thetapsp, phiss, B_r, L)
            rssp= rssp - learning_rate * gradient
            phiss = spiral(phipsp, rpsp, rssp, vresult)
            loss_history[p] = J(rssp, thetapsp, phiss, B_r, L)
            grad[p] = gradient
            para[p]=rssp
            print(gradient,rssp, loss_history)
        return rssp, loss_history,para,grad


    rssp, loss_history,para,gradient = gradient_descent(rssp, thetapsp, phiss, B_r, L, learning_rate, num_iters)
    Iter=np.array([i+1 for i in range(num_iters)])

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'savefig.dpi': 300,

    })

    colors = [
        '#5D7B8C',
        '#8C756D',
        '#6B8E23'
    ]

    fig = plt.figure(figsize=(5.3, 4.3))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(Iter, para, color=colors[0], linewidth=2.0, label='Parameter $R_{ss}$')
    ax1.set_xlabel('Iter (N)', color='k', fontweight='bold')
    ax1.set_ylabel('Heliocentric distance ($R_{s}$)', color=colors[0], fontweight='bold')
    ax1.tick_params(axis='y', colors=colors[0])
    ax1.spines['left'].set_color(colors[0])
    ax2 = ax1.twinx()
    ax2.plot(Iter,loss_history,'--',color=colors[1],linewidth=2.0,  label='MSE')
    ax2.spines['right'].set_position(('outward', 20))
    ax2.set_ylabel('MSE', color=colors[1],fontweight='bold')
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.spines['right'].set_color(colors[1])
    ax3 = ax1.twinx()
    ax3.plot(Iter, gradient, '--', color=colors[2], linewidth=2.0, label='Gradient')
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Gradient', color=colors[2], fontweight='bold')
    ax3.tick_params(axis='y', colors=colors[2])
    ax3.spines['right'].set_color(colors[2])
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=2)
        ax.tick_params(which='major', length=4)
    lines = [ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.15),
               frameon=False,
               ncol=3)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='#D3D3D3')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax3.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax3.get_xticklabels():
        label.set_fontweight('bold')

    ax_scatter = fig.add_subplot(2, 1, 2)
    #魈#colors = [
        #'#75C44E','#297245','#C2CFA2','#5EA69C', '#316658', '#333C42',
    # '#A4799E',
    #]
    #宵宫
    colors = ['#D2BF9F', '#EB9505', '#5D74A2', '#EFCF36', '#A4799E']
    np.random.seed(42)
    x_base = np.linspace(0, 10, 50)
    idex=np.array([0,2,4,8,num_iters-1])
    Brpsp=std*Brpsp+mean
    for i in range(6):
        if i==0:
            ax_scatter.scatter(times[Brpsp > 0], Brpsp[Brpsp > 0], color='#E64A19', s=0.7, linewidths=0.7, label='Positive $B_r$')
            ax_scatter.scatter(times[Brpsp < 0], Brpsp[Brpsp < 0], color='#1976D2', s=0.7, linewidths=0.7, label='Negative $B_r$')
            phiss = spiral(phipsp, rpsp, 2.5, vresult)
            Brs = Bss(2.5, thetapsp, phiss, B_r, L) * 2.5 * 2.5 / (rpsp ** 2) * 1e+5
            Brs = np.interp(np.arange(len(Brs)), np.flatnonzero(~np.isnan(Brs)), Brs[~np.isnan(Brs)])

            ax_scatter.plot(times, Brs, '-', c='black', linewidth=2.2, label='$2.5R_s$')

        else:
            k=idex[i-1]
            L = 22
            phiss = spiral(phipsp, rpsp, para[k], vresult)
            Brs = Bss(para[k], thetapsp, phiss, B_r, L) * para[k] * para[k] / (rpsp ** 2) * 1e+5
            Brs = np.interp(np.arange(len(Brs)), np.flatnonzero(~np.isnan(Brs)), Brs[~np.isnan(Brs)])

            ax_scatter.plot(times, Brs, '--', c=colors[i - 1], linewidth=1.5, label=f'Iter {idex[i - 1] + 1}')
    ax.axhline(0, color='black', linewidth=0.5)
    ax_scatter.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_scatter.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_scatter.tick_params(which='both', direction='in', top=True, right=True)
    ax_scatter.tick_params(which='minor', length=2)
    ax_scatter.tick_params(which='major', length=4)
    ax_scatter.set_xlabel('Date' , labelpad=2, fontweight='bold')
    ax_scatter.set_ylabel(r'$\bf{B_r}$ / nT', labelpad=2, fontweight='bold')
    ax_scatter.yaxis.set_major_locator(MaxNLocator(prune='both'))
    def custom_format(value, tick_number):
        return f"{value:.1f}"
    ax_scatter.yaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f"{value:.1f}"))
    date_format = mdates.DateFormatter('%y-%m-%d')
    ax_scatter.xaxis.set_major_formatter(date_format)
    ax_scatter.tick_params(axis='x', rotation=45, pad=2)
    handles, labels = ax_scatter.get_legend_handles_labels()
    for label in ax_scatter.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax_scatter.get_xticklabels():
        label.set_fontweight('bold')
    fig.legend(handles, labels,
               loc='upper right',
               bbox_to_anchor=(0.93, 0.45),
               frameon=False,
               ncol=1,
               columnspacing=1.2,
               handletextpad=0.3)
    ax_scatter.grid(True, which='major', linestyle=':', linewidth=0.6, color='#E0E0E0')
    plt.tight_layout()
    plt.savefig('Optimal-12.png', format='png', bbox_inches='tight')
    plt.show()