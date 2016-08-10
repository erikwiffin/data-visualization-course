---
layout: post
title: "Assignment: Running an analysis of variance"
---

## Program and outputs

```python
import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

data = pandas.read_csv('gapminder.csv')
print('Total number of countries: {0}'.format(len(data)))
```

    Total number of countries: 213



```python
# Convert numeric types
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors='coerce')
data['internetuserate'] = pandas.to_numeric(data['internetuserate'], errors='coerce')

sub1 = data[['incomeperperson', 'internetuserate']].dropna()

print('Remaining number of countries: {0}'.format(len(sub1)))

# Since GDP per person isn't categorical data, I'm going to group it by magnitude first
groups = [pow(10, i) for i in range(2, 7)]
labels = ['{0} - {1}'.format(groups[index], i) for index, i in enumerate(groups[1:])]
sub1['incomeperperson'] = pandas.cut(sub1['incomeperperson'], groups, right=False, labels=labels)
```

    Remaining number of countries: 183



```python
model = smf.ols(formula='internetuserate ~ C(incomeperperson)', data=sub1).fit()
print(model.summary())
print('\n'*2)

print('Means for internet use rate by income per person')
m = sub1.groupby('incomeperperson').mean()
print(m)
print('\n'*2)

print('standard deviations for internet use rate by income per person')
sd = sub1.groupby('incomeperperson').std()
print(sd)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        internetuserate   R-squared:                       0.677
    Model:                            OLS   Adj. R-squared:                  0.673
    Method:                 Least Squares   F-statistic:                     188.3
    Date:                Tue, 09 Aug 2016   Prob (F-statistic):           7.59e-45
    Time:                        21:02:32   Log-Likelihood:                -765.55
    No. Observations:                 183   AIC:                             1537.
    Df Residuals:                     180   BIC:                             1547.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==========================================================================================================
                                                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
    ----------------------------------------------------------------------------------------------------------
    Intercept                                  8.3944      2.219      3.783      0.000         4.016    12.773
    C(incomeperperson)[T.1000 - 10000]        24.2198      2.811      8.616      0.000        18.673    29.766
    C(incomeperperson)[T.10000 - 100000]      62.8513      3.258     19.292      0.000        56.423    69.280
    C(incomeperperson)[T.100000 - 1000000]          0          0        nan        nan             0         0
    ==============================================================================
    Omnibus:                        3.852   Durbin-Watson:                   1.985
    Prob(Omnibus):                  0.146   Jarque-Bera (JB):                3.473
    Skew:                           0.326   Prob(JB):                        0.176
    Kurtosis:                       3.175   Cond. No.                          inf
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is      0. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    
    
    
    Means for internet use rate by income per person
                      internetuserate
    incomeperperson                  
    100 - 1000               8.394409
    1000 - 10000            32.614232
    10000 - 100000          71.245728
    100000 - 1000000              NaN
    
    
    
    standard deviations for internet use rate by income per person
                      internetuserate
    incomeperperson                  
    100 - 1000               8.328892
    1000 - 10000            19.402752
    10000 - 100000          15.484270
    100000 - 1000000              NaN



```python
 mc = multi.MultiComparison(sub1['internetuserate'], sub1['incomeperperson'])
 res = mc.tukeyhsd()
 print(res.summary())
```

         Multiple Comparison of Means - Tukey HSD,FWER=0.05    
    ===========================================================
       group1        group2     meandiff  lower   upper  reject
    -----------------------------------------------------------
     100 - 1000   1000 - 10000  24.2198  17.5765 30.8631  True 
     100 - 1000  10000 - 100000 62.8513  55.1516  70.551  True 
    1000 - 10000 10000 - 100000 38.6315  31.6736 45.5894  True 
    -----------------------------------------------------------



## Summary

A p-value of 7.59e-45 seems pretty significant, and in all comparisons, I was able to reject the null hypothesis. It seems certain that income per person has a very strong predictive effect on internet use rate.
