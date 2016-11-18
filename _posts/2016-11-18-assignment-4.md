---
layout: post
title: "Assignment: Test a Logistic Regression Model"
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
data['polityscore'] = pandas.to_numeric(data['polityscore'], errors='coerce')
data['internetuserate'] = pandas.to_numeric(data['internetuserate'], errors='coerce')
data['employrate'] = pandas.to_numeric(data['employrate'], errors='coerce')
data['urbanrate'] = pandas.to_numeric(data['urbanrate'], errors='coerce')
data['armedforcesrate'] = pandas.to_numeric(data['armedforcesrate'], errors='coerce')

# Since there are no categorical variables in the gapminder dataset,
# I'm creating one by grouping polity scores less than vs greater than 0.
labels = [0, 1]
data['polityscore_bins'] = pandas.cut(sub1['polityscore'], bins=2, labels=labels)
data['polityscore_bins'] = pandas.to_numeric(data['polityscore_bins'], errors='coerce')

sub1 = data[['polityscore', 'polityscore_bins', 'internetuserate', 'employrate', 'urbanrate', 'armedforcesrate']].dropna()

print('Total remaining countries: {0}'.format(len(sub1)))
```

    Total remaining countries: 148


### Logistic regression with no confounding variables



```python
lreg1 = smf.logit(formula='polityscore_bins ~ internetuserate', data=sub1).fit()
lreg1.summary()
```





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th> <td>polityscore_bins</td> <th>  No. Observations:  </th>  <td>   148</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   146</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 18 Nov 2016</td> <th>  Pseudo R-squ.:     </th>  <td>0.06948</td> 
</tr>
<tr>
  <th>Time:</th>              <td>17:03:41</td>     <th>  Log-Likelihood:    </th> <td> -86.076</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -92.503</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>0.0003367</td>
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>       <td>    0.0138</td> <td>    0.271</td> <td>    0.051</td> <td> 0.959</td> <td>   -0.517     0.545</td>
</tr>
<tr>
  <th>internetuserate</th> <td>    0.0255</td> <td>    0.008</td> <td>    3.298</td> <td> 0.001</td> <td>    0.010     0.041</td>
</tr>
</table>




```python
def odds_ratios(lreg):
    params = lreg.params
    conf = lreg.conf_int()
    conf['OR'] = params
    conf.columns = ['Lower CI', 'Upper CI', 'OR']
    
    return np.exp(conf)

odds_ratios(lreg1)
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lower CI</th>
      <th>Upper CI</th>
      <th>OR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>0.596194</td>
      <td>1.724212</td>
      <td>1.013886</td>
    </tr>
    <tr>
      <th>internetuserate</th>
      <td>1.010417</td>
      <td>1.041558</td>
      <td>1.025869</td>
    </tr>
  </tbody>
</table>
</div>


### Logistic regression with confounding variables



```python
lreg2 = smf.logit(formula='polityscore_bins ~ internetuserate + employrate + urbanrate + armedforcesrate', data=sub1).fit()
lreg2.summary()
```





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th> <td>polityscore_bins</td> <th>  No. Observations:  </th>  <td>   148</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   143</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     4</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 18 Nov 2016</td> <th>  Pseudo R-squ.:     </th>  <td>0.1908</td>  
</tr>
<tr>
  <th>Time:</th>              <td>17:03:47</td>     <th>  Log-Likelihood:    </th> <td> -74.857</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -92.503</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>4.046e-07</td>
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>       <td>    4.2600</td> <td>    1.615</td> <td>    2.638</td> <td> 0.008</td> <td>    1.095     7.425</td>
</tr>
<tr>
  <th>internetuserate</th> <td>    0.0364</td> <td>    0.012</td> <td>    3.015</td> <td> 0.003</td> <td>    0.013     0.060</td>
</tr>
<tr>
  <th>employrate</th>      <td>   -0.0515</td> <td>    0.022</td> <td>   -2.371</td> <td> 0.018</td> <td>   -0.094    -0.009</td>
</tr>
<tr>
  <th>urbanrate</th>       <td>   -0.0094</td> <td>    0.013</td> <td>   -0.750</td> <td> 0.453</td> <td>   -0.034     0.015</td>
</tr>
<tr>
  <th>armedforcesrate</th> <td>   -0.6771</td> <td>    0.185</td> <td>   -3.651</td> <td> 0.000</td> <td>   -1.041    -0.314</td>
</tr>
</table>




```python
odds_ratios(lreg2)
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lower CI</th>
      <th>Upper CI</th>
      <th>OR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>2.990108</td>
      <td>1677.028382</td>
      <td>70.813105</td>
    </tr>
    <tr>
      <th>internetuserate</th>
      <td>1.012822</td>
      <td>1.061937</td>
      <td>1.037088</td>
    </tr>
    <tr>
      <th>employrate</th>
      <td>0.910171</td>
      <td>0.991097</td>
      <td>0.949772</td>
    </tr>
    <tr>
      <th>urbanrate</th>
      <td>0.966505</td>
      <td>1.015320</td>
      <td>0.990612</td>
    </tr>
    <tr>
      <th>armedforcesrate</th>
      <td>0.353227</td>
      <td>0.730799</td>
      <td>0.508072</td>
    </tr>
  </tbody>
</table>
</div>

## Summary

My hypothesis was that internet use rate would be fairly well correlated with a nation&rsquo;s polity score.

In my first logistic regression, I was surprised to find a high significance, but very low odds ratio (OR=1.03, 95% CI = 1.01-1.04, p=.001). Apparently, my hypothesis was incorrect.

After adjusting for potential confounding factors (employment rate, urbanrate, and armed forces rate), the odds of having a polity score greater than zero were almost half as likely for countries with a higher armed forces rate (OR=0.51, 95% CI = 0.35-0.73, p<0.001). None of the other factors were significantly associated with the polity score.
