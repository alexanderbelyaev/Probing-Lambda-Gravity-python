from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import random

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

delR=1E-3#M
M=0.1#kg
G=6.67430E-11#N m^2/kg^2
L=1.1056E-52 #m^-2
da_omega=0.48E-17
c=3E+08#m/s^2 (speed of light)

def func(R, G, L):
    return (2*delR/R**2)*(M*G/R + L*R**2*c**2/6)
    
xdata = np.linspace(0.07, 0.12, 6)
y = func(xdata, G, L)

I=0
delG_acc=0
delL_acc=0
L_acc=0

while I<1000:
	rng_tmp = np.random.default_rng()
	y_noise_tmp = da_omega*rng_tmp.normal(size=xdata.size)
	ydata_tmp = y + y_noise_tmp
	popt_tmp, pcov_tmp = curve_fit(func, xdata, ydata_tmp)
	errors_tmp=np.sqrt(np.diag(pcov_tmp))

	if not (np.isfinite(errors_tmp).any()):
		continue
	
	delG_tmp=errors_tmp[0]/G
	delL_tmp=errors_tmp[1]/L
	L_tmp=abs(popt_tmp[1])
	delG_acc=delG_acc+delG_tmp
	delL_acc=delL_acc+delL_tmp
	L_acc=L_acc+L_tmp
	I +=1
	
L_av=L_acc/I
delL_av=delL_acc/I
delG_av=delG_acc/I

print('L_avg,delL_avg,del_G,I=',L_av,delG_av,I)
print('=====================')

	

rng = np.random.default_rng()
y_noise = da_omega*rng.normal(size=xdata.size)

#y1 = da_omega*rng.normal(size=1000000)
#y2 = rng.normal(0.,da_omega,size=1000000)
#count1, bins1, _ = plt.hist(y1, 100, density=True)
#count2, bins2, _ = plt.hist(y2, 100, density=True)

#plt.plot(bins1, ,linewidth=2, color='r')
#plt.plot(bins2, y2,linewidth=2, color='y')
plt.show()

print(y_noise)
ydata = y + y_noise



popt, pcov = curve_fit(func, xdata, ydata)
print('popt=',popt)
print('pcov=',pcov)
errors=np.sqrt(np.diag(pcov))
print('errors=',errors)
delG=errors[0]/G
delL=errors[1]/L

#delG=abs(popt[0]-G)/G
#delL=abs(popt[1]-L)/L
print('delG=',delG)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=[5,5])
plt.subplots_adjust(top=0.95,left=0.15,right=0.99)

#np.array([2.56274217, 1.37268521, 0.47427475])
plt.subplot(211)
plt.plot(xdata, ydata, 'o', label='data',color='b')
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit:')
#plt.plot([], [], ' ', label='$G=$%5.5e N m$^2$/kg$^2$ \n ${\\Delta G/G=}$%5.1e   \n ${\\Lambda<}$%5.1e 1/m$^2$ ' % tuple([popt[0],delG,abs(popt[1])]))
plt.plot([], [], ' ', label='$G=$%5.5e N m$^2$/kg$^2$ \n ${\\Delta G/G=}$%5.1e   \n ${\\Lambda<}$%5.1e 1/m$^2$ ' % tuple([popt[0],delG_av,L_av]))
 

plt.legend()
plt.ylabel('$a$[m/s$^2$]')


plt.subplot(212)
dy=(ydata-func(xdata, *popt))/func(xdata, *popt)
dybec=da_omega/y
plt.plot(xdata, dy, 'o',color='b',label='$(a^{\\rm{exp}}- a^{\\rm{th}})/a^{\\rm{th}}$')
plt.errorbar(xdata, dy, yerr=dybec,color='b', fmt='o', capsize=3)

plt.plot(xdata, -dybec , '--',color='b')
plt.plot(xdata, dybec ,  '--', label='$\\Delta a^{\\rm{BEC}}/a^{\\rm{th}}$',color='b')
#plt.plot([0.07,0.12], [0,0] ,  '--', color='black',lw=1)
plt.axhline(y=0, color='black', linestyle='-',lw=0.5)

plt.legend()
#plt.ylabel('$(a_\\Omega^{\\rm exp}- a_\\Omega^{\\rm th})/a_\\Omega^{\\rm th}$')
plt.ylim(-1.5E-6,1.5E-6)

#print('==',tuple(popt))
print('==',tuple([popt[0],popt[1],delG,delL]))
	 
plt.xlabel('R[m]')

#plt.show()
plt.savefig("fit.pdf")
	 
