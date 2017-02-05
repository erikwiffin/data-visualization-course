---
layout: post
title: "Assignment: Running a k-means Cluster Analysis"
---

## Program and outputs

### Data loading and cleaning


```python
import pandas as pd

CSV_PATH = 'gapminder.csv'

data = pd.read_csv(CSV_PATH)
print('Total number of countries: {0}'.format(len(data)))
```

    Total number of countries: 213



```python
import numpy as np

PREDICTORS = [
    'incomeperperson', 'alcconsumption', 'armedforcesrate',
    'breastcancerper100th', 'co2emissions', 'femaleemployrate',
    'hivrate', 'internetuserate',
    'polityscore', 'relectricperperson', 'suicideper100th',
    'employrate', 'urbanrate'
]

clean = data.copy()

clean = clean.replace(r'\s+', np.NaN, regex=True)
clean = clean[PREDICTORS + ['lifeexpectancy']].dropna()

for key in PREDICTORS + ['lifeexpectancy']:
    clean[key] = pd.to_numeric(clean[key], errors='coerce')


clean = clean.dropna()

print('Countries remaining:', len(clean))
clean.head()
```

    Countries remaining: 107





<div style="overflow: auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>incomeperperson</th>
      <th>alcconsumption</th>
      <th>armedforcesrate</th>
      <th>breastcancerper100th</th>
      <th>co2emissions</th>
      <th>femaleemployrate</th>
      <th>hivrate</th>
      <th>internetuserate</th>
      <th>polityscore</th>
      <th>relectricperperson</th>
      <th>suicideper100th</th>
      <th>employrate</th>
      <th>urbanrate</th>
      <th>lifeexpectancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2231.993335</td>
      <td>0.69</td>
      <td>2.306817</td>
      <td>23.5</td>
      <td>2.932109e+09</td>
      <td>31.700001</td>
      <td>0.1</td>
      <td>12.500073</td>
      <td>2</td>
      <td>590.509814</td>
      <td>4.848770</td>
      <td>50.500000</td>
      <td>65.22</td>
      <td>73.131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1381.004268</td>
      <td>5.57</td>
      <td>1.461329</td>
      <td>23.1</td>
      <td>2.483580e+08</td>
      <td>69.400002</td>
      <td>2.0</td>
      <td>9.999954</td>
      <td>-2</td>
      <td>172.999227</td>
      <td>14.554677</td>
      <td>75.699997</td>
      <td>56.70</td>
      <td>51.093</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10749.419238</td>
      <td>9.35</td>
      <td>0.560987</td>
      <td>73.9</td>
      <td>5.872119e+09</td>
      <td>45.900002</td>
      <td>0.5</td>
      <td>36.000335</td>
      <td>8</td>
      <td>768.428300</td>
      <td>7.765584</td>
      <td>58.400002</td>
      <td>92.00</td>
      <td>75.901</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1326.741757</td>
      <td>13.66</td>
      <td>2.618438</td>
      <td>51.6</td>
      <td>5.121967e+07</td>
      <td>34.200001</td>
      <td>0.1</td>
      <td>44.001025</td>
      <td>5</td>
      <td>603.763058</td>
      <td>3.741588</td>
      <td>40.099998</td>
      <td>63.86</td>
      <td>74.241</td>
    </tr>
    <tr>
      <th>9</th>
      <td>25249.986061</td>
      <td>10.21</td>
      <td>0.486280</td>
      <td>83.2</td>
      <td>1.297009e+10</td>
      <td>54.599998</td>
      <td>0.1</td>
      <td>75.895654</td>
      <td>10</td>
      <td>2825.391095</td>
      <td>8.470030</td>
      <td>61.500000</td>
      <td>88.74</td>
      <td>81.907</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

predictors = clean[PREDICTORS].copy()
for key in PREDICTORS:
    predictors[key] = preprocessing.scale(predictors[key])

# split data into train and test sets
clus_train, clus_test = train_test_split(predictors, test_size=.3, random_state=123)
```

### Running a k-means Cluster Analysis


```python
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# k-means cluster analysis for 1-9 clusters                                                           
clusters = range(1,10)
meandist = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) / clus_train.shape[0])
```


```python
import matplotlib.pyplot as plt

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()
```


![png]({{ site.baseurl }}/public/images/2017-02-03/output_4_0.png)



```python
%matplotlib inline

from sklearn.decomposition import PCA

models = {}

# plot clusters
def plot(clus_train, model, n):
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(clus_train)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model.labels_,)
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    plt.title('Scatterplot of Canonical Variables for {} Clusters'.format(n))
    plt.show()
    
for n in range(2, 6):
    # Interpret N cluster solution
    models[n] = KMeans(n_clusters=n)
    models[n].fit(clus_train)
    clusassign = models[n].predict(clus_train)
    
    plot(clus_train, models[n], n)
```


| ![png]({{ site.baseurl }}/public/images/2017-02-03/output_5_0.png) | ![png]({{ site.baseurl }}/public/images/2017-02-03/output_5_1.png) |
| ![png]({{ site.baseurl }}/public/images/2017-02-03/output_5_2.png) | ![png]({{ site.baseurl }}/public/images/2017-02-03/output_5_3.png) |



```python
"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)

# create a list that has the new index variable
cluslist = list(clus_train['index'])

# create a list of cluster assignments
labels = list(models[3].labels_)

# combine index variable list with cluster assignment list into a dictionary
newlist = dict(zip(cluslist, labels))

# convert newlist dictionary to a dataframe
newclus = pd.DataFrame.from_dict(newlist, orient='index')

# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)

# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train = pd.merge(clus_train, newclus, on='index')

# cluster frequencies
merged_train.cluster.value_counts()

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
```

#### Clustering variable means by cluster


```python
clustergrp = merged_train.groupby('cluster').mean()
clustergrp
```



<div style="overflow: auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>cluster</th>
      <th>level_0</th>
      <th>index</th>
      <th>incomeperperson</th>
      <th>alcconsumption</th>
      <th>armedforcesrate</th>
      <th>breastcancerper100th</th>
      <th>co2emissions</th>
      <th>femaleemployrate</th>
      <th>hivrate</th>
      <th>internetuserate</th>
      <th>polityscore</th>
      <th>relectricperperson</th>
      <th>suicideper100th</th>
      <th>employrate</th>
      <th>urbanrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31.450000</td>
      <td>99.150000</td>
      <td>-0.686916</td>
      <td>-0.697225</td>
      <td>0.013628</td>
      <td>-0.755176</td>
      <td>-0.240625</td>
      <td>0.931895</td>
      <td>0.372560</td>
      <td>-0.957167</td>
      <td>-0.641475</td>
      <td>-0.647425</td>
      <td>-0.317349</td>
      <td>1.040447</td>
      <td>-0.882169</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41.615385</td>
      <td>121.923077</td>
      <td>1.995773</td>
      <td>0.310344</td>
      <td>-0.165620</td>
      <td>1.510134</td>
      <td>0.857540</td>
      <td>0.479623</td>
      <td>-0.354374</td>
      <td>1.528231</td>
      <td>0.568169</td>
      <td>1.881545</td>
      <td>0.000761</td>
      <td>0.450520</td>
      <td>0.964292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37.341463</td>
      <td>108.902439</td>
      <td>-0.300487</td>
      <td>0.184597</td>
      <td>0.150248</td>
      <td>-0.104142</td>
      <td>-0.126604</td>
      <td>-0.668625</td>
      <td>0.059176</td>
      <td>-0.049347</td>
      <td>0.147707</td>
      <td>-0.221579</td>
      <td>-0.096890</td>
      <td>-0.704371</td>
      <td>0.206884</td>
    </tr>
  </tbody>
</table>
</div>



```python
# validate clusters in training data by examining cluster differences in life expectancy using ANOVA
# first have to merge life expectancy with clustering variables and cluster assignment data 
le_data = clean['lifeexpectancy']

# split GPA data into train and test sets
le_train, le_test = train_test_split(le_data, test_size=.3, random_state=123)
le_train1 = pd.DataFrame(le_train)
le_train1.reset_index(level=0, inplace=True)
merged_train_all = pd.merge(le_train1, merged_train, on='index')
sub1 = merged_train_all[['lifeexpectancy', 'cluster']].dropna()
```


```python
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

lemod = smf.ols(formula='lifeexpectancy ~ C(cluster)', data=sub1).fit()
lemod.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>lifeexpectancy</td>  <th>  R-squared:         </th> <td>   0.471</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.457</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   31.67</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Feb 2017</td> <th>  Prob (F-statistic):</th> <td>1.47e-10</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:40:56</td>     <th>  Log-Likelihood:    </th> <td> -243.38</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    74</td>      <th>  AIC:               </th> <td>   492.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    71</td>      <th>  BIC:               </th> <td>   499.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>       <td>   62.3610</td> <td>    1.481</td> <td>   42.101</td> <td> 0.000</td> <td>   59.408    65.314</td>
</tr>
<tr>
  <th>C(cluster)[T.1]</th> <td>   18.1989</td> <td>    2.360</td> <td>    7.712</td> <td> 0.000</td> <td>   13.493    22.905</td>
</tr>
<tr>
  <th>C(cluster)[T.2]</th> <td>   10.2167</td> <td>    1.807</td> <td>    5.655</td> <td> 0.000</td> <td>    6.614    13.819</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.030</td> <th>  Durbin-Watson:     </th> <td>   1.942</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.004</td> <th>  Jarque-Bera (JB):  </th> <td>  11.189</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.837</td> <th>  Prob(JB):          </th> <td> 0.00372</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.908</td> <th>  Cond. No.          </th> <td>    4.44</td>
</tr>
</table>


#### Means for Life Expectancy by cluster

```python
sub1.groupby('cluster').mean()
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>cluster</th>
      <th>lifeexpectancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62.361000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.559923</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72.577732</td>
    </tr>
  </tbody>
</table>
</div>


#### Standard deviations for Life Expectancy by cluster

```python
sub1.groupby('cluster').std()
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>cluster</th>
      <th>lifeexpectancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.717742</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.450612</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.415368</td>
    </tr>
  </tbody>
</table>
</div>




```python
mc1 = multi.MultiComparison(sub1['lifeexpectancy'], sub1['cluster'])
res1 = mc1.tukeyhsd()
res1.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD,FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>   <th>lower</th>   <th>upper</th>  <th>reject</th>
</tr>
<tr>
     <td>0</td>      <td>1</td>    <td>18.1989</td>  <td>12.5495</td> <td>23.8483</td>  <td>True</td> 
</tr>
<tr>
     <td>0</td>      <td>2</td>    <td>10.2167</td>  <td>5.8917</td>  <td>14.5418</td>  <td>True</td> 
</tr>
<tr>
     <td>1</td>      <td>2</td>    <td>-7.9822</td> <td>-13.0296</td> <td>-2.9348</td>  <td>True</td> 
</tr>
</table>

## Summary

Based on the plots of clusters based on Canonical variables, I proceeded with 3 clusters as my model. My three clusters were as follows:

* Cluster 0 - This looks like developing nations facing severe poverty. The urban rate is the lowest, combined with low income per person, and a high employment rate, looks like argicultural economies where everyone needs to work in order to remain at substinance income levels.
* Cluster 1 - This is very clearly a cluster of developed nations. There is a significantly higher incomeperperson, and higher consumption of things like CO2, residential electricity, and internet. There is also a much higher urban rate and polity score.
* Cluster 2 - This cluster is more similar to cluster 0, but with a trend towards urbanization. Income per person is higher, consumption is up, and the employment rate is way down.

There was a difference of about 10 years of mean life expectancy between each of my clusters, and the Tukey test considered each cluster as significant.
