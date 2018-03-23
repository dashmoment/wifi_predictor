import seaborn as sns
import matplotlib.pyplot as plt

def check_correlation(data):
    
    corrmat = data.corr()
    plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    
    print(corrmat['Delay-mean'].sort_values(ascending=False))
    print(corrmat['Delay-mean_log'].sort_values(ascending=False))
    print(corrmat['Delay-max'].sort_values(ascending=False))
    print(corrmat['Delay-max_log'].sort_values(ascending=False))