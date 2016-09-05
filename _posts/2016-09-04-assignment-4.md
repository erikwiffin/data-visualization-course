---
layout: post
title: "Assignment: Testing a Potential Moderator"
---

## Program and outputs

```python
%matplotlib inline
import pandas
import numpy
import scipy
import seaborn
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

data = pandas.read_csv('gapminder.csv')
print('Total number of countries: {0}'.format(len(data)))
```

    Total number of countries: 213

```python
# Convert numeric types
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors='coerce')
data['alcconsumption'] = pandas.to_numeric(data['alcconsumption'], errors='coerce')
data['lifeexpectancy'] = pandas.to_numeric(data['lifeexpectancy'], errors='coerce')

clean = data[['incomeperperson', 'alcconsumption', 'lifeexpectancy']].dropna()
print('Total remaining countries: {0}'.format(len(clean)))
```

    Total remaining countries: 171

```python
sub1 = clean[(clean['lifeexpectancy'] <= 70)]
sub2 = clean[(clean['lifeexpectancy'] > 70)]

def plot(data):
    seaborn.regplot(x="incomeperperson", y="alcconsumption", data=data)
    plt.xlabel('Income per Person')
    plt.ylabel('Alcohol Consumption')

print('Association between income and alcohol consumption for those with a SHORTER life expectancy')
print(('correlation coefficient', 'p value'))
print(scipy.stats.pearsonr(sub1['incomeperperson'], sub1['alcconsumption']))
plot(sub1)

print('Association between income and alcohol consumption for those with a LONGER life expectancy')
print(('correlation coefficient', 'p value'))
print(scipy.stats.pearsonr(sub2['incomeperperson'], sub2['alcconsumption']))
plot(sub2)
```

    Association between income and alcohol consumption for those with a SHORTER life expectancy
    ('correlation coefficient', 'p value')
    (0.24564412558074125, 0.038937397029065977)

![png]({{ site.baseurl }}/public/images/2016-09-04/output_4_0.png)

    Association between income and alcohol consumption for those with a LONGER life expectancy
    ('correlation coefficient', 'p value')
    (0.20396295528103475, 0.04180701551814929)

![png]({{ site.baseurl }}/public/images/2016-09-04/output_5_0.png)

## Summary

I ran a pearsonr correlation for income and alcohol consumption, moderated by countries with a SHORT (less than 70 years) life expectancy or a LONG (longer than 70 years) life expectancy. Despite having a significant P-value, the correlation coefficient is low enough that we can say that there is no correlation between income per person and alcohol consumption for either subgroup.
