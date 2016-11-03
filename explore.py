
import pandas as pd
import numpy as np


import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt


train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv") 


train.head()



df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
  

X = train.iloc[:,:-1]
y = train.TARGET

X['numZeros'] = (X==0).sum(axis=1)
train['numZeros'] = X['numZeros']

 g = sns.FacetGrid(train, hue="TARGET", size=5)
 g.map(sns.kdeplot,  "var15")
 g.add_legend()

 plt.show()
 