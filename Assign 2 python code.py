# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:23:37 2022

@author: Shiva Sheelvanth
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

x=pd.Series([24.23, 25.53, 25.41, 24.14, 29.62, 28.25, 25.81, 24.39, 40.26, 32.95, 91.36, 25.99, 39.42, 26.71, 35.00])

name=('Allied Signal', 'Bankers Trust', 'General Mills', 'ITT Industries', 'J.P.Morgan & Co.', 'Lehman Brothers', 'Marriott', 'MCI', 'Merrill Lynch', 'Microsoft', 
'Morgan Stanley', 'Sun Microsystems', 'Travelers', 'US Airways', 'Warner-Lambert')

plt.figure(figsize=(6,8))
plt.pie(x,labels=name,autopct='%1.0f%%')
plt.show()

sns.boxplot(x)