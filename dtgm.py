# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import partial_dependence
dt = pd.read_excel('Desktop/statistics/dt111.xlsx')
#dt = dt1[dt1['weekday']==1]
min_max_scaler = preprocessing.MinMaxScaler()
dt.iloc[:,6:] = min_max_scaler.fit_transform(dt.iloc[:,6:])

for j in range(0,629):
    dt.iloc[j,5]=math.log(dt.iloc[j,5])
    for i in range(6,33):
        dt.iloc[j,i]=math.log(dt.iloc[j,i]+1)
        #print(j,i)
y= dt.iloc[:,5]
x= dt.iloc[:,6:]

x = sm.add_constant(x) 
model = sm.OLS(y, x).fit() 
print(model.summary()) 
x = dt.iloc[:,6:]

x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=0)
for h in range(1,50):
   
    dtree = DecisionTreeRegressor(max_depth=h)
    dtree.fit(x1,y1)
    print(dtree.score(x2,y2),h)
print(dtree.score(x,y))
dtree = DecisionTreeRegressor(max_depth=44)
dtree.fit(x1,y1)
r = permutation_importance(dtree, x, y,n_repeats=30,random_state=0)

dt1 = pd.read_excel('Desktop/statistics/dt111.xlsx')
pdp1 = partial_dependence(dtree, x, [26])
xv = dt1.iloc[:,32]

v = xv.drop_duplicates()
v.to_excel('/Desktop/pdpv.xlsx')


import matplotlib.pyplot as plt

pdp = pd.read_excel('/Desktop/statistics/pdp.xlsx')
a = pdp.iloc[:,42].values
b = pdp.iloc[:,43].values*100
print(pdp.columns[20])
plt.figure(figsize=(5, 5), dpi=100)

plt.plot(a,b,'o-')

plt.show()