---
layout: post
title: "Assignment: Running a Lasso Regression Analysis"
---

## Program and outputs

### Data loading and cleaning


```python
import pandas as pandas
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pylab as plt

CSV_PATH = 'gapminder.csv'

data = pandas.read_csv(CSV_PATH)
print('Total number of countries: {0}'.format(len(data)))
```

    Total number of countries: 213



```python
PREDICTORS = [
    'incomeperperson', 'alcconsumption', 'armedforcesrate',
    'breastcancerper100th', 'co2emissions', 'femaleemployrate',
    'hivrate', 'internetuserate',
    'polityscore', 'relectricperperson', 'suicideper100th',
    'employrate', 'urbanrate'
]

clean = data.copy()
for key in PREDICTORS + ['lifeexpectancy']:
    clean[key] = pandas.to_numeric(clean[key], errors='coerce')

clean = clean.dropna()

print('Countries remaining:', len(clean))
clean.head()
```

    Countries remaining: 107





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>incomeperperson</th>
      <th>alcconsumption</th>
      <th>armedforcesrate</th>
      <th>breastcancerper100th</th>
      <th>co2emissions</th>
      <th>femaleemployrate</th>
      <th>hivrate</th>
      <th>internetuserate</th>
      <th>lifeexpectancy</th>
      <th>oilperperson</th>
      <th>polityscore</th>
      <th>relectricperperson</th>
      <th>suicideper100th</th>
      <th>employrate</th>
      <th>urbanrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>2231.993335</td>
      <td>0.69</td>
      <td>2.306817</td>
      <td>23.5</td>
      <td>2.932109e+09</td>
      <td>31.700001</td>
      <td>0.1</td>
      <td>12.500073</td>
      <td>73.131</td>
      <td>.42009452521537</td>
      <td>2.0</td>
      <td>590.509814</td>
      <td>4.848770</td>
      <td>50.500000</td>
      <td>65.22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>1381.004268</td>
      <td>5.57</td>
      <td>1.461329</td>
      <td>23.1</td>
      <td>2.483580e+08</td>
      <td>69.400002</td>
      <td>2.0</td>
      <td>9.999954</td>
      <td>51.093</td>
      <td></td>
      <td>-2.0</td>
      <td>172.999227</td>
      <td>14.554677</td>
      <td>75.699997</td>
      <td>56.70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Argentina</td>
      <td>10749.419238</td>
      <td>9.35</td>
      <td>0.560987</td>
      <td>73.9</td>
      <td>5.872119e+09</td>
      <td>45.900002</td>
      <td>0.5</td>
      <td>36.000335</td>
      <td>75.901</td>
      <td>.635943800978195</td>
      <td>8.0</td>
      <td>768.428300</td>
      <td>7.765584</td>
      <td>58.400002</td>
      <td>92.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Armenia</td>
      <td>1326.741757</td>
      <td>13.66</td>
      <td>2.618438</td>
      <td>51.6</td>
      <td>5.121967e+07</td>
      <td>34.200001</td>
      <td>0.1</td>
      <td>44.001025</td>
      <td>74.241</td>
      <td></td>
      <td>5.0</td>
      <td>603.763058</td>
      <td>3.741588</td>
      <td>40.099998</td>
      <td>63.86</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Australia</td>
      <td>25249.986061</td>
      <td>10.21</td>
      <td>0.486280</td>
      <td>83.2</td>
      <td>1.297009e+10</td>
      <td>54.599998</td>
      <td>0.1</td>
      <td>75.895654</td>
      <td>81.907</td>
      <td>1.91302610912404</td>
      <td>10.0</td>
      <td>2825.391095</td>
      <td>8.470030</td>
      <td>61.500000</td>
      <td>88.74</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import preprocessing

predictors = clean[PREDICTORS].copy()
for key in PREDICTORS:
    predictors[key] = preprocessing.scale(predictors[key])
    
targets = clean.lifeexpectancy
    
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.3, random_state=123)

print(pred_train.shape, pred_test.shape, tar_train.shape, tar_test.shape)
```

    (74, 13) (33, 13) (74,) (33,)




```python
from collections import OrderedDict
from sklearn.linear_model import LassoLarsCV

model = LassoLarsCV(cv=10, precompute=False).fit(pred_train, tar_train)

OrderedDict(sorted(zip(predictors.columns, model.coef_), key=lambda x:x[1], reverse=True))
```




    OrderedDict([('internetuserate', 2.9741932507050883),
                 ('incomeperperson', 1.5624998619776493),
                 ('polityscore', 0.95348158080473089),
                 ('urbanrate', 0.62824156642092388),
                 ('alcconsumption', 0.0),
                 ('armedforcesrate', 0.0),
                 ('breastcancerper100th', 0.0),
                 ('relectricperperson', 0.0),
                 ('co2emissions', -0.065710252825883983),
                 ('femaleemployrate', -0.16966106864470906),
                 ('suicideper100th', -0.83797198915263182),
                 ('employrate', -1.3086675757200679),
                 ('hivrate', -3.6033945847485298)])




```python
# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.show()
```


![png]({{ site.baseurl }}/public/images/2017-01-25/output_4_0.png)



```python
# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.show()
```



![png]({{ site.baseurl }}/public/images/2017-01-25/output_5_1.png)



```python
# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print('training data MSE')
print(train_error)
print('test data MSE')
print(test_error)
```

    training data MSE
    14.0227968412
    test data MSE
    22.9565114677



```python
# R-square from training and test data
rsquared_train = model.score(pred_train, tar_train)
rsquared_test = model.score(pred_test, tar_test)
print('training data R-square')
print(rsquared_train)
print('test data R-square')
print(rsquared_test)
```

    training data R-square
    0.823964900718
    test data R-square
    0.658213145158



```python
from collections import OrderedDict
from sklearn.linear_model import LassoLarsCV

model2 = LassoLarsCV(cv=10, precompute=False).fit(predictors, targets)

print('mse', mean_squared_error(targets, model2.predict(predictors)))
print('r-square', model2.score(predictors, targets))

OrderedDict(sorted(zip(predictors.columns, model2.coef_), key=lambda x:x[1], reverse=True))
```

    mse 17.7754276093
    r-square 0.766001466082

    OrderedDict([('internetuserate', 2.6765897850358265),
                 ('incomeperperson', 1.4881319407059432),
                 ('urbanrate', 0.62065826306013672),
                 ('polityscore', 0.49665728486271465),
                 ('alcconsumption', 0.0),
                 ('armedforcesrate', 0.0),
                 ('breastcancerper100th', 0.0),
                 ('co2emissions', 0.0),
                 ('femaleemployrate', 0.0),
                 ('relectricperperson', 0.0),
                 ('suicideper100th', 0.0),
                 ('employrate', -0.86922466889577155),
                 ('hivrate', -3.6439368063365305)])


## Summary

After running the Lasso regression, my model showed that HIV rate (-3.6) and internet use rate (3.0) were the most influential features in determining a country&rsquo;s life expectancy. My model resulted in an R-square of 0.66 when run against the test dataset, down from 0.82 against the training set. This is a noticeable drop, but still high enough to suggest that these are reliable features.

Alcohol consumption, the armed forces rate, incidences of breast cancer, and residential electricity consumption ended up being reduced out of the model.

When I re-ran the model against the entire dataset (with ~100 records, the split dataset is incredibly small), it resulted in an R-square of 0.77, with the same features coming out on top. However, CO2 emissions, female employment rate, and sucide rates were all removed from the model.

Lasso regression seems like an incredibly useful tool to use at the start of data analysis, to identify features that are likely to produce useful results under other analysis methods.
