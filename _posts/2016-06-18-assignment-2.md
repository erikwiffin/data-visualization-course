---
layout: post
title: "Assignment: Running Your First Program"
---

## Program and outputs

```python
import pandas
import numpy

data = pandas.read_csv('gapminder.csv')
print('Total number of countries: {0}'.format(len(data)))
```

    Total number of countries: 213



```python
# Convert numeric types and drop NaNs
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors='coerce')
data['incomeperperson'].dropna(inplace=True)

print('Remaining number of countries: {0}'.format(len(data['incomeperperson'])))
```

    Remaining number of countries: 190



```python
# Since GDP per person isn't categorical data, I'm going to group it by magnitude first
groups = [pow(10, i) for i in range(2, 7)]
labels = ['{0} - {1}'.format(groups[index], i) for index, i in enumerate(groups[1:])]
print('Groups: {0}'.format(labels))
```

    Groups: ['100 - 1000', '1000 - 10000', '10000 - 100000', '100000 - 1000000']



```python
grouped = pandas.cut(data['incomeperperson'], groups, right=False, labels=labels)
print('Counts for GDP per person, grouped by magnitude:')
print(grouped.value_counts(sort=False))
print('\nPercentages for GDP per person, grouped by magnitude:')
print(grouped.value_counts(sort=False, normalize=True))
```

    Counts for GDP per person, grouped by magnitude:
    100 - 1000          54
    1000 - 10000        89
    10000 - 100000      46
    100000 - 1000000     1
    Name: incomeperperson, dtype: int64
    
    Percentages for GDP per person, grouped by magnitude:
    100 - 1000          0.284211
    1000 - 10000        0.468421
    10000 - 100000      0.242105
    100000 - 1000000    0.005263
    Name: incomeperperson, dtype: float64



```python
# Now do the above for all of my consumption types
types = [
    ('alcconsumption', 'Alcohol Consumption'),
    ('co2emissions', 'CO2 Emissions'),
    ('internetuserate', 'Internet Use Rate'),
    ('oilperperson', 'Oil per Person'),
    ('relectricperperson', 'Electricity per Person'),
]
def summarize(series, name):
    # Convert to numeric and drop NaNs
    series = pandas.to_numeric(series, errors='coerce')
    series.dropna(inplace=True)

    percentiles = numpy.linspace(0, 1, 5)
    groups = list(series.quantile(percentiles))
    labels = ['{0} - {1}'.format(groups[index], i) for index, i in enumerate(groups[1:])]
    grouped = pandas.cut(series, groups, right=False, labels=labels)
    
    print(name)
    print('-' * len(name))
    
    print('Counts for {0} grouped by percentile:'.format(name))
    print(grouped.value_counts(sort=False))
    
    print('Percentages for {0}, grouped by percentile (should probably be 25%)'.format(name))
    print(grouped.value_counts(sort=False, normalize=True))

for (key, name) in types:
    summarize(data[key], name)
    print('\n')
```

    Alcohol Consumption
    -------------------
    Counts for Alcohol Consumption grouped by percentile:
    0.03 - 2.625     47
    2.625 - 5.92     45
    5.92 - 9.925     48
    9.925 - 23.01    46
    Name: alcconsumption, dtype: int64
    Percentages for Alcohol Consumption, grouped by percentile (should probably be 25%)
    0.03 - 2.625     0.252688
    2.625 - 5.92     0.241935
    5.92 - 9.925     0.258065
    9.925 - 23.01    0.247312
    Name: alcconsumption, dtype: float64
    
    
    CO2 Emissions
    -------------
    Counts for CO2 Emissions grouped by percentile:
    132000.0 - 34846166.66666667             50
    34846166.66666667 - 185901833.3333335    50
    185901833.3333335 - 1846084166.666665    50
    1846084166.666665 - 334220872333.333     49
    Name: co2emissions, dtype: int64
    Percentages for CO2 Emissions, grouped by percentile (should probably be 25%)
    132000.0 - 34846166.66666667             0.251256
    34846166.66666667 - 185901833.3333335    0.251256
    185901833.3333335 - 1846084166.666665    0.251256
    1846084166.666665 - 334220872333.333     0.246231
    Name: co2emissions, dtype: float64
    
    
    Internet Use Rate
    -----------------
    Counts for Internet Use Rate grouped by percentile:
    0.210066325622776 - 9.999603951038267    48
    9.999603951038267 - 31.81012075468915    48
    31.81012075468915 - 56.41604586287351    48
    56.41604586287351 - 95.6381132075472     47
    Name: internetuserate, dtype: int64
    Percentages for Internet Use Rate, grouped by percentile (should probably be 25%)
    0.210066325622776 - 9.999603951038267    0.251309
    9.999603951038267 - 31.81012075468915    0.251309
    31.81012075468915 - 56.41604586287351    0.251309
    56.41604586287351 - 95.6381132075472     0.246073
    Name: internetuserate, dtype: float64
    
    
    Oil per Person
    --------------
    Counts for Oil per Person grouped by percentile:
    0.03228146619272 - 0.5325414918259135      16
    0.5325414918259135 - 1.03246988375935      15
    1.03246988375935 - 1.6227370046323601      16
    1.6227370046323601 - 12.228644991426199    15
    Name: oilperperson, dtype: int64
    Percentages for Oil per Person, grouped by percentile (should probably be 25%)
    0.03228146619272 - 0.5325414918259135      0.258065
    0.5325414918259135 - 1.03246988375935      0.241935
    1.03246988375935 - 1.6227370046323601      0.258065
    1.6227370046323601 - 12.228644991426199    0.241935
    Name: oilperperson, dtype: float64
    
    
    Electricity per Person
    ----------------------
    Counts for Electricity per Person grouped by percentile:
    0.0 - 203.65210850945525                  34
    203.65210850945525 - 597.1364359554304    34
    597.1364359554304 - 1491.145248925905     34
    1491.145248925905 - 11154.7550328078      33
    Name: relectricperperson, dtype: int64
    Percentages for Electricity per Person, grouped by percentile (should probably be 25%)
    0.0 - 203.65210850945525                  0.251852
    203.65210850945525 - 597.1364359554304    0.251852
    597.1364359554304 - 1491.145248925905     0.251852
    1491.145248925905 - 11154.7550328078      0.244444
    Name: relectricperperson, dtype: float64
    
    
## Conclusions

Since my data wasn&rsquo;t categorical, it was a bit tricky to make it all work with value counts. However, that gave me the opportunity to learn a bit more about the pandas library, and how to create categories out of non-categorical data.

One thing I noticed pretty quickly, was that for both GDP and my various consumption variables, once the values started growing, they grow very quickly. For example, most of the alcohol consumption variables centered around 5 liters, but at the high end, it went as far as 23 liters.
