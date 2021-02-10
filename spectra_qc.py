import numpy as np
import os
import matplotlib.pyplot as plt

# files = os.listdir("spectra")

# for filename in files:
#     np.genfromtxt(filename, delimiter=",")


spectrum = np.genfromtxt("spectra/E (1).TXT", delimiter=",")
spectrum = np.transpose(spectrum)

plt.plot(spectrum[0], spectrum[1])
plt.show()