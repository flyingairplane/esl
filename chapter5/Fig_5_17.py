##figure 5.17
##2020/09/25
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import seaborn as sns
import pywt

wavelet = pywt.Wavelet('sym8')
phi, psi,x= wavelet.wavefun(level=5)


fig, axes = plt.subplots(1,2)

axes[0].plot(np.arange(0,len(phi)),phi,marker='*')
axes[1].plot(np.arange(0,len(psi)),psi,marker='*')
plt.show()