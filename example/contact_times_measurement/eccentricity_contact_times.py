from wasp39b_params import period_day
from karate.calc_contact_times import calc_contact_times
import numpy as np
import matplotlib.pyplot as plt

# WASP-39 b
rp_over_rs = 0.145
period = period_day * 24 * 60 * 60
a_over_rs = 11.4
cosi = 0.45 / a_over_rs
t0 = 0


ecosw = np.linspace(0, 0.5, 10)
ecosw = np.concatenate([ecosw, -ecosw[1:]])
esinw = 0
ecc = np.sqrt(ecosw**2 + esinw**2)
omega = np.arctan2(0, ecosw[1:] / ecc[1:])
omega = np.insert(omega, 0, np.arctan2(1, 0))
t1, t2, t3, t4 = calc_contact_times(rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0)
fig = plt.figure(figsize=(12, 6))
plt.plot(t1, ecosw, marker=".", linestyle="None", label="t1")
plt.plot(t2, ecosw, marker=".", linestyle="None", label="t2")
plt.plot(t3, ecosw, marker=".", linestyle="None", label="t3")
plt.plot(t4, ecosw, marker=".", linestyle="None", label="t4")
plt.vlines(
    [t1[0], t2[0], t3[0], t4[0]],
    ymin=[-0.5, -0.5, -0.5, -0.5],
    ymax=[0.5, 0.5, 0.5, 0.5],
    colors="black",
)
plt.legend()
plt.title(r"contact times with ecos$\omega$")
plt.xlabel("Time")
plt.ylabel(r"ecos$\omega$")
plt.savefig("contact_times_cosw.png")
plt.close()


ecosw = 0
esinw = np.linspace(0, 0.5, 20)
esinw = np.concatenate([esinw, -esinw[1:]])
ecc = np.sqrt(ecosw**2 + esinw**2)
omega = np.arctan2(esinw[1:] / ecc[1:], 0)
omega = np.insert(omega, 0, np.arctan2(1, 0))
t1, t2, t3, t4 = calc_contact_times(rp_over_rs, period, a_over_rs, ecc, omega, cosi, t0)
fig = plt.figure(figsize=(12, 6))
plt.plot(t1, esinw, marker=".", linestyle="None", label="t1")
plt.plot(t2, esinw, marker=".", linestyle="None", label="t2")
plt.plot(t3, esinw, marker=".", linestyle="None", label="t3")
plt.plot(t4, esinw, marker=".", linestyle="None", label="t4")
plt.vlines(
    [t1[0], t2[0], t3[0], t4[0]],
    ymin=[-0.5, -0.5, -0.5, -0.5],
    ymax=[0.5, 0.5, 0.5, 0.5],
    colors="black",
)
plt.legend()
plt.title(r"contact times with esin$\omega$")
plt.xlabel("Time")
plt.ylabel(r"esin$\omega$")
plt.savefig("contact_times_sinw.png")
plt.close()
