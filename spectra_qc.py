import numpy as np
import os
import matplotlib.pyplot as plt

# files = os.listdir("spectra")

# for filename in files:
#     np.genfromtxt(filename, delimiter=",")


spectrum = np.genfromtxt("spectra/E (1).TXT", delimiter=",")
spectrum = np.transpose(spectrum)

fig, ax = plt.subplots()
ax.plot()