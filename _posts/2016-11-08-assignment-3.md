---
layout: post
title: "Assignment: Test a Multiple Regression Model"
---

## Program and outputs

### Data loading and cleaning

```python
import numpy as np
import pandas as pandas
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

data = pandas.read_csv('gapminder.csv')
print('Total number of countries: {0}'.format(len(data)))
```

    Total number of countries: 213



```python
data['alcconsumption'] = pandas.to_numeric(data['alcconsumption'], errors='coerce')
data['femaleemployrate'] = pandas.to_numeric(data['femaleemployrate'], errors='coerce')
data['employrate'] = pandas.to_numeric(data['employrate'], errors='coerce')

sub1 = data[['alcconsumption', 'femaleemployrate', 'employrate']].dropna()
print('Total remaining countries: {0}'.format(len(sub1)))
```

    Total remaining countries: 162



```python
# Center quantitative IVs for regression analysis
sub1['femaleemployrate_c'] = sub1['femaleemployrate'] - sub1['femaleemployrate'].mean()
sub1['employrate_c'] = sub1['employrate'] - sub1['employrate'].mean()
sub1[['femaleemployrate_c', 'employrate_c']].mean()
```




    femaleemployrate_c    2.434267e-15
    employrate_c          3.640435e-15
    dtype: float64


### Single regression


```python
# First order (linear) scatterplot
scat1 = seaborn.regplot(x='femaleemployrate', y='alcconsumption', scatter=True, data=sub1)

# Second order (polynomial) scatterplot
scat1 = seaborn.regplot(x='femaleemployrate', y='alcconsumption', scatter=True, order=2, data=sub1)
plt.xlabel('Employed percentage of female population')
plt.ylabel('Alcohol consumption per adult')
plt.show()
```


![png]({{ site.baseurl }}/public/images/2016-11-08/output_3_0.png)




```python
# Linear regression analysis
reg1 = smf.ols('alcconsumption ~ femaleemployrate_c', data=sub1).fit()
reg1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>alcconsumption</td>  <th>  R-squared:         </th> <td>   0.029</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.023</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4.787</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 08 Nov 2016</td> <th>  Prob (F-statistic):</th>  <td>0.0301</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>11:20:44</td>     <th>  Log-Likelihood:    </th> <td> -487.57</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   162</td>      <th>  AIC:               </th> <td>   979.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   160</td>      <th>  BIC:               </th> <td>   985.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>          <td>    6.8124</td> <td>    0.388</td> <td>   17.559</td> <td> 0.000</td> <td>    6.046     7.579</td>
</tr>
<tr>
  <th>femaleemployrate_c</th> <td>    0.0578</td> <td>    0.026</td> <td>    2.188</td> <td> 0.030</td> <td>    0.006     0.110</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12.411</td> <th>  Durbin-Watson:     </th> <td>   1.915</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.002</td> <th>  Jarque-Bera (JB):  </th> <td>  13.772</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.713</td> <th>  Prob(JB):          </th> <td> 0.00102</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.909</td> <th>  Cond. No.          </th> <td>    14.7</td>
</tr>
</table>


```python
# Quadratic (polynomial) regression analysis
reg2 = smf.ols('alcconsumption ~ femaleemployrate_c + I(femaleemployrate_c**2)', data=sub1).fit()
reg2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>alcconsumption</td>  <th>  R-squared:         </th> <td>   0.135</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.125</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   12.45</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 08 Nov 2016</td> <th>  Prob (F-statistic):</th> <td>9.45e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:20:45</td>     <th>  Log-Likelihood:    </th> <td> -478.17</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   162</td>      <th>  AIC:               </th> <td>   962.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   159</td>      <th>  BIC:               </th> <td>   971.6</td>
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
               <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>                  <td>    7.9540</td> <td>    0.449</td> <td>   17.720</td> <td> 0.000</td> <td>    7.067     8.840</td>
</tr>
<tr>
  <th>femaleemployrate_c</th>         <td>    0.0605</td> <td>    0.025</td> <td>    2.419</td> <td> 0.017</td> <td>    0.011     0.110</td>
</tr>
<tr>
  <th>I(femaleemployrate_c ** 2)</th> <td>   -0.0053</td> <td>    0.001</td> <td>   -4.423</td> <td> 0.000</td> <td>   -0.008    -0.003</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.022</td> <th>  Durbin-Watson:     </th> <td>   1.930</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.011</td> <th>  Jarque-Bera (JB):  </th> <td>   9.369</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.589</td> <th>  Prob(JB):          </th> <td> 0.00924</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.024</td> <th>  Cond. No.          </th> <td>    459.</td>
</tr>
</table>



### Multiple Regression


```python
# Controlling for total employment rate
reg3 = smf.ols('alcconsumption ~ femaleemployrate_c + I(femaleemployrate_c**2) + employrate_c', data=sub1).fit()
reg3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>alcconsumption</td>  <th>  R-squared:         </th> <td>   0.345</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.333</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   27.79</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 08 Nov 2016</td> <th>  Prob (F-statistic):</th> <td>1.73e-14</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:20:46</td>     <th>  Log-Likelihood:    </th> <td> -455.63</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   162</td>      <th>  AIC:               </th> <td>   919.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   158</td>      <th>  BIC:               </th> <td>   931.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>                  <td>    7.5561</td> <td>    0.396</td> <td>   19.092</td> <td> 0.000</td> <td>    6.774     8.338</td>
</tr>
<tr>
  <th>femaleemployrate_c</th>         <td>    0.3312</td> <td>    0.044</td> <td>    7.553</td> <td> 0.000</td> <td>    0.245     0.418</td>
</tr>
<tr>
  <th>I(femaleemployrate_c ** 2)</th> <td>   -0.0034</td> <td>    0.001</td> <td>   -3.204</td> <td> 0.002</td> <td>   -0.006    -0.001</td>
</tr>
<tr>
  <th>employrate_c</th>               <td>   -0.4481</td> <td>    0.063</td> <td>   -7.119</td> <td> 0.000</td> <td>   -0.572    -0.324</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.239</td> <th>  Durbin-Watson:     </th> <td>   1.862</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.326</td> <th>  Jarque-Bera (JB):  </th> <td>   1.788</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.205</td> <th>  Prob(JB):          </th> <td>   0.409</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.311</td> <th>  Cond. No.          </th> <td>    463.</td>
</tr>
</table>




```python
#Q-Q plot for normality
sm.qqplot(reg3.resid, line='r')
plt.show()

# simple plot of residuals
stdres = pandas.DataFrame(reg3.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
plt.show()

# additional regression diagnostic plots
fig2 = plt.figure(figsize=(12,8))
sm.graphics.plot_regress_exog(reg3,  "employrate_c", fig=fig2)
plt.show()

# leverage plot
sm.graphics.influence_plot(reg3, size=8)
plt.show()
```


![png]({{ site.baseurl }}/public/images/2016-11-08/output_8_0.png)



![png]({{ site.baseurl }}/public/images/2016-11-08/output_8_1.png)



![png]({{ site.baseurl }}/public/images/2016-11-08/output_8_2.png)



![png]({{ site.baseurl }}/public/images/2016-11-08/output_8_3.png)


## Summary

I started by analyzing the effect on female employment rate on alcohol consumption per adult. I found a significant (p = 0.0301) positive (coef = 0.0578) effect that explained a very small (r-squared = 0.029) portion of the data.

I then added a polynomial regression for female employment rate. I found a more significant (p = 9.45e-06) effect that explained a larger (r-squared = 0.135) portion of the data. Both of the variables individually are significant (p = 0.017 and p < 0.001) which confirms that the best fit line includes some curvature.

Assuming that the general employment rate would have an effect on the female employment rate, I decided to run my multiple regression, correcting for total employment rate. Again, my results were more significant (p = 1.73e-14) and had a greater effect (r-squared = 0.345). Again, all my variables were individually significant (p < 0.001, p = 0.002, p < 0.001).

Despite the overall significance of my explanatory variables, the total effect size was only 35%, so there are a fair number of residuals outside of 1 and 2 standard deviations. Nonetheless, the residuals with the greatest leverage were within 2 standard deviations, so I am confident saying that female employment rate, independent of the total employment rate, has a significant effect on alcohol consumption.
