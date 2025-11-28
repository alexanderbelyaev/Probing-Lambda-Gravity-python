from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import random

xm=25
ym=10

def f(x, y):
  if (abs(x) <=xm and abs(y) <=ym ):
    return 0.5+random.random()/(ym+xm)
  else:
    if (x==0):
      alp=np.arccos(-1)
    else:
      alp=np.arctan((abs(y)/abs(x)))
      
    if (alp < np.arctan((abs(ym)/abs(xm))) ):
      ls=xm/np.cos(alp)
    else:
      ls=ym/np.sin(alp)
      
    var=(np.sqrt((x**2+y**2))-ls)**2*(1.-0.4*random.random())
    return np.exp(-var/(0.3*ls**2))

vf=np.vectorize(f)

  



fig = plt.figure(figsize=[3,3])
ax = plt.axes(projection='3d')
ax.view_init(elev=45, azim=60, roll=0)

x = np.linspace(-50, 50, 52)
y = np.linspace(-50, 50, 52)
X, Y = np.meshgrid(x, y)
Z = vf(X, Y)
#ax.contour3D(X, Y, Z, 50, cmap='binary')
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
ax.plot_surface(X, Y, Z,  cmap='jet', edgecolor='none')

#xr=np.random.random(10000)
#yr=np.random.random(10000)
#x = xr*100-50
#y = yr*100-50
#z = vf(x, y)
#ax.scatter(x, y, z, c=z, cmap='jet', linewidth=0.5);

ax.set_xlabel('$\\mu m$')
ax.set_ylabel('$\\mu m$')
ax.set_zticks([])

plt.savefig("bec-3d.pdf")
