import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import label_generator_rahul as label_gen_r

import pickle
from sklearn.feature_selection import SelectKBest, f_regression


# load data
with open('../data/raw_data.pkl', 'rb') as input:

    train = pickle.load(input)
    label_train = pickle.load(input)
    test = pickle.load(input)
    label_test = pickle.load(input)


#print(train.head(1), '\n')
#print(label_train['delay_mean'].head(1), '\n')

label_train_class = label_train['delay_mean'].loc[:,'Delay-mean']
label_train_class = label_gen_r.transform_delay2category(label_train_class)
print(label_train_class.head(3))
print(type(label_train_class))
print(label_train_class.value_counts())

'''
total = pd.concat([train, test])
label_total = pd.concat([label_train['delay_mean'], label_test['delay_mean']])
label_total = pd.DataFrame(label_total).rename(columns={'Delay-mean': 'delay_mean'})

train = total.sample(frac=0.9)
sample_idx = train.index
label_train = label_total.iloc[sample_idx]

test = total[~total.index.isin(sample_idx)]
label_test = label_total[~total.index.isin(sample_idx)]

# feature selection
anova_filter = SelectKBest(f_regression, k=4).fit(
    train, label_train['delay_mean'])
mask = anova_filter.get_support(indices=True)
print(mask)
print(list(train.columns.values))
'''


'''
output = label_gen.transform_delay2category(pd.DataFrame(label_train['delay_mean']))
output.rename('delay_mean', inplace = True)
output = pd.DataFrame(output)
output_onehot = pd.get_dummies(output['delay_mean'])
print(output_onehot)
'''
