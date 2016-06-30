
---
layout: post
title: "Assignment: Making Data Management Decisions"
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
# Convert numeric types
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors='coerce')

# Drop any countries with missing GPD
data = data[pandas.notnull(data['incomeperperson'])]

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

    grouped = pandas.qcut(series, 4)
    
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
    [0.05, 2.73]       45
    (2.73, 6.12]       45
    (6.12, 10.035]     44
    (10.035, 23.01]    45
    Name: alcconsumption, dtype: int64
    Percentages for Alcohol Consumption, grouped by percentile (should probably be 25%)
    [0.05, 2.73]       0.251397
    (2.73, 6.12]       0.251397
    (6.12, 10.035]     0.245810
    (10.035, 23.01]    0.251397
    Name: alcconsumption, dtype: float64
    
    
    CO2 Emissions
    -------------
    Counts for CO2 Emissions grouped by percentile:
    [850666.667, 39924500]            45
    (39924500, 234864666.667]         45
    (234864666.667, 2138961000]       44
    (2138961000, 334220872333.333]    45
    Name: co2emissions, dtype: int64
    Percentages for CO2 Emissions, grouped by percentile (should probably be 25%)
    [850666.667, 39924500]            0.251397
    (39924500, 234864666.667]         0.251397
    (234864666.667, 2138961000]       0.245810
    (2138961000, 334220872333.333]    0.251397
    Name: co2emissions, dtype: float64
    
    
    Internet Use Rate
    -----------------
    Counts for Internet Use Rate grouped by percentile:
    [0.21, 9.949]         46
    (9.949, 31.00438]     46
    (31.00438, 55.646]    45
    (55.646, 95.638]      46
    Name: internetuserate, dtype: int64
    Percentages for Internet Use Rate, grouped by percentile (should probably be 25%)
    [0.21, 9.949]         0.251366
    (9.949, 31.00438]     0.251366
    (31.00438, 55.646]    0.245902
    (55.646, 95.638]      0.251366
    Name: internetuserate, dtype: float64
    
    
    Oil per Person
    --------------
    Counts for Oil per Person grouped by percentile:
    [0.0323, 0.505]    16
    (0.505, 0.891]     15
    (0.891, 1.593]     15
    (1.593, 12.229]    15
    Name: oilperperson, dtype: int64
    Percentages for Oil per Person, grouped by percentile (should probably be 25%)
    [0.0323, 0.505]    0.262295
    (0.505, 0.891]     0.245902
    (0.891, 1.593]     0.245902
    (1.593, 12.229]    0.245902
    Name: oilperperson, dtype: float64
    
    
    Electricity per Person
    ----------------------
    Counts for Electricity per Person grouped by percentile:
    [0, 226.318]             33
    (226.318, 609.335]       32
    (609.335, 1484.703]      32
    (1484.703, 11154.755]    33
    Name: relectricperperson, dtype: int64
    Percentages for Electricity per Person, grouped by percentile (should probably be 25%)
    [0, 226.318]             0.253846
    (226.318, 609.335]       0.246154
    (609.335, 1484.703]      0.246154
    (1484.703, 11154.755]    0.253846
    Name: relectricperperson, dtype: float64

## Summary

I began by dropping any rows where GDP per capita was missing, since I&rsquo;m looking to eventually compare that to my various consumption categories. Then I showed the output of GDP grouped by magnitude (100 - 1000, 1000 - 10000, 10000 - 100000, 100000 - 1000000) since those gave me the most meaningful groupings.

Once that was done, I wanted to summarize each of my consumption types. For those I used the `qcut` method explained in this weeks lesson to break them each into quartiles.

Alcohol, CO2, and Internet use rate generally seem to have pretty thorough coverage of data (~180 countries), but oil and electricity per person are missing a lot more data. I&rsquo;ll probably end up just using the first three types.
