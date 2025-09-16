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
rssp=2.500
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
dt=20
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

            Brss = Bss(2.0, Y, X, B_r, 25)
            X1 = X * 180 / np.pi
            Y1 = Y * 180 / np.pi

            ax = plt.subplot(gs[i, j])
            ax.invert_yaxis()
            c = ax.pcolormesh(X1, Y1, Brss, cmap='RdBu_r', vmin=-np.max(np.abs(Brss)), vmax=np.max(np.abs(Brss)))
            cbar = plt.colorbar(c)
            contour_zero =ax.contour(X1, Y1, Brss, levels=[0.00], colors='black', linestyles='solid')

            cbar.ax.tick_params(labelsize=7)
            ax.clabel(contour_zero, inline=True, fontsize=6)
            cbar.ax.set_xlabel('$B_{R_{ss}}$', labelpad=5, rotation=0)
            ax.set_title('Field Extrapolation on $R_{ss}$ (Encounter' + sequence[k] + ')', fontsize=8)
            ax.set_xlabel('Longitude Coordinate,$\phi$', fontsize=7)
            ax.set_ylabel('colatitude Coordinate,$\\theta$', fontsize=7)
            if j >= 1:
                ax.set_ylabel('')
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='minor', length=2)
            ax.tick_params(which='major', length=4)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
            ax.xaxis.set_major_locator(MaxNLocator(prune='both'))
            ax.tick_params(axis='x', which='major', labelsize=7)
            ax.tick_params(axis='y', which='major', labelsize=7)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.savefig('Source Surface.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

