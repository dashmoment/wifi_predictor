import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr  


h5data = pd.HDFStore('../data/scan_data_1061129.h5')
raw_data= h5data["raw_data"]

data = np.array(raw_data["Spectralscan_data"])
label =  raw_data["Spectralscan_mean"]

feature_wise = []

for i in range(len(data[0])):
    
    tmp = []
    for j in range(len(data)):
        tmp.append(data[j][i])
        
    feature_wise.append(tmp)
        
con_relate = []
p_value = []

for i in range(len(feature_wise)):
    cof, pval = pearsonr(feature_wise[i], feature_wise[30])
    con_relate.append(cof)
    p_value.append(pval)


x, y = zip(*sorted(zip(feature_wise[10], feature_wise[1])))

plt.plot(con_relate)
plt.show()
plt.plot(p_value)
plt.show()