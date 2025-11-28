from mpl_toolkits import mplot3d
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
matplotlib.use("Agg")  # For non-interactive, file-only output (like saving PNGs)

hbar=1.054571817E-34 #J*s = kg*m/s
pi=np.arccos(-1)
alpha=np.array([0.3,0.15,0.05])
L=np.array([150,500,1000])*1E-06 #m
l=2
n=1
m=1.44E-25 #kg
N0=1E9
theta=0.31
tau=60.*24.*3600. #seconds in 30 days
rb=5.291772105E-11#m, Bohr radius
a_sl=99.*rb #Rb 99 scattering length
Nd=1
Nph=1100
ts=1
c=3E+08#m/s^2 (speed of light)
da_om_num=alpha*hbar*pi**3*np.sqrt(2*n*l)*(l**2-n**2)**2
da_om_den=16.*m*N0*theta*np.sqrt(L*a_sl*tau*ts*Nph*Nd)*(l**2+n**2)
delta_a_omega=da_om_num/da_om_den #masetr formula for Delta a_Omega
print('delta_a_omega=',delta_a_omega)

# Additional parameters #
n0=np.array([1E14*1E06])
c_s=(4.*pi*99*rb*n0)**0.5*hbar/m #m*kg*m/s/kg= 
Omega=pi*c_s*(n+l)/L

print ('c_s=',c_s)
print ('a_sl=',a_sl)
print ('Omega=',Omega)


R0=0.1#m
M=0.1#kg
delta_R=0.001#m
dG=delta_a_omega*R0**3/(2*M)/delta_R
dL=delta_a_omega*3/delta_R/c**2

print('delta G=',dG)
print('delta Lambda=',dL)


# --- Sweep over Nph values and plot delta_a_omega vs Nph in dB ---

Nph_values = np.logspace(0, 3.3, 200) #
Nph_dB = 10 * np.log10(Nph_values)   # Convert to dB

delta_a_omega_array = []

for Nph_i in Nph_values:
    da_om_num = alpha * hbar * pi**3 * np.sqrt(2 * n * l) * (l**2 - n**2)**2
    da_om_den = 16. * m * N0 * theta * np.sqrt(L * a_sl * tau * ts * Nph_i * Nd) * (l**2 + n**2)
    delta_a_omega_i = da_om_num / da_om_den
    delta_a_omega_array.append(delta_a_omega_i)

delta_a_omega_array = np.array(delta_a_omega_array)

# Plotting with y-axis in units of 1e-18 m/s^2
plt.figure(figsize=(6, 4))
for i in range(3):  # Plot for each of the 3 alpha/L values
    plt.plot(Nph_dB, delta_a_omega_array[:, i] * 1e18, label = fr"$L = {L[i]*1e6:.0f}\,\mu\mathrm{{m}},\,\alpha={alpha[i]}$")

plt.yscale('log')
plt.xlabel(r"$N_{\mathrm{p}}$ (dB)", fontsize=12)
plt.ylabel(r"$\Delta a^\text{BEC} \,\, [10^{-18} \, \mathrm{m/s}^2]$", fontsize=12)
#plt.title(r"$\Delta a^\text{BEC}$ vs $N_{\mathrm{p}}$ in dB", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("delta_a_vs_Nph_dB.pdf")
print(delta_a_omega_array[:, 2] * 1e18)



# Plotting with y-axis in units of 1e-18 m/s^2
plt.figure(figsize=(6, 4))
for i in range(3):  # Plot for each of the 3 alpha/L values
    label = fr"$L = {L[i]*1e6:.0f}\,\mu\mathrm{{m}},\,\alpha={alpha[i]}$"
    plt.plot(Nph_dB, delta_a_omega_array[:, i] * 1e18, label=label)

plt.yscale('log')
plt.xlabel(r"$N_{\mathrm{p}}$ (dB)", fontsize=12)
plt.ylabel(r"$\Delta a^\text{BEC} \,\, [10^{-18} \, \mathrm{m/s}^2]$", fontsize=12)
#plt.title(r"$\Delta a^\text{BEC}$ vs $N_{\mathrm{p}}$ in dB", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("delta_a_vs_Nph_dB.pdf")
print(delta_a_omega_array[:, 2] * 1e18)


plt.figure(figsize=(6, 4))
for i in range(3):  # One curve per L and alpha
    plt.plot(Nph_values, delta_a_omega_array[:, i] * 1e18,
             label = fr"$L = {L[i]*1e6:.0f}\,\mu\mathrm{{m}},\,\alpha={alpha[i]}$")

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$N_{\mathrm{p}}$", fontsize=12)
plt.ylabel(r"$\Delta a^\text{BEC} \,\, [10^{-18} \, \mathrm{m/s}^2]$", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("delta_a_vs_Nph_logx.pdf")

plt.show()

