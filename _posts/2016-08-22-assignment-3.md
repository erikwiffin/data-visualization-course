---
layout: post
title: "Assignment: Generating a Correlation Coefficient"
---

## Program and outputs

```python
%matplotlib inline
import pandas
import numpy
import scipy
import seaborn
import matplotlib.pyplot as plt

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
```


```python
scat1 = seaborn.regplot(x='incomeperperson', y='lifeexpectancy', fit_reg=True, data=clean, logx=True)
plt.xlabel('Income per Person')
plt.ylabel('Life Expectancy')
plt.title('Scatterplot for the Association between Income per Person and Life Expectancy')

print('Association between Income per Person and Life Expectancy')
print(scipy.stats.pearsonr(clean['incomeperperson'], clean['lifeexpectancy']))
```

![png]({{ site.baseurl }}/public/images/2016-08-22/output_2_1.png)

    Association between Income per Person and Life Expectancy
    (0.59534257610705288, 8.874239655396052e-18)


```python
scat2 = seaborn.regplot(x='alcconsumption', y='lifeexpectancy', fit_reg=True, data=clean)
plt.xlabel('Alcohol Consumption')
plt.ylabel('Life Expectancy')
plt.title('Scatterplot for the Association between Alcohol Consumption and Life Expectancy')

print('Association between Alcohol Consumption and Life Expectancy')
print(scipy.stats.pearsonr(clean['alcconsumption'], clean['lifeexpectancy']))
```

![png]({{ site.baseurl }}/public/images/2016-08-22/output_3_1.png)

    Association between Alcohol Consumption and Life Expectancy
    (0.29893173996914729, 7.1401036474160216e-05)


## Summary

There is a significant difference between my two graphs, and the second one definitely doesn't look significant.

There is a strong correlation between Income per Person and Life Expectancy. This has a very low p value, so we can say that this is significant.

There is a much weaker correlation between Alcohol Consumption and Life Expectancy, but the p value is still fairly low, so at least we can be confident about it.
