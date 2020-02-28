---
layout: post
title: 천체 데이터 분류_PART(1)_데이터시각화
tags: [Python, 분류, 예측]
description: DACON 천체 예측하기
comments: true
---
<br/>
머신러닝을 통해 분류 연습을 해보고 싶어 데이터를 찾던 중, DACON에서 진행했던 '천체 유형 분류'를 찾았다. 
모르고 대회 기간을 놓쳐 결국 파일을 제출하지 못했지만, 그 아쉬움을 코드 정리하면서 위안을 얻어보려고 한다. 
천체 데이터는 아래 링크에서 확인할 수 있으니 참고하도록 하자.  
https://www.dacon.io/competitions/official/235573
<br/>
분석의 전체적인 흐름은 다음과 같이 3개의 PART로 정리할 예정이다. 
<br/>
<br/>

|  PART  |  내용  |
| --------- | :--- | :--------- | :----- |
| PART1 | 1) Prepare Problem<br/>1.1) load libraries<br/> 1.2) load and explore the shape of the dataset <br/><br/>  2) Summarize Data<br/>2.1) Descriptive statistics<br/>2.2) Visualization |
| PART2 | 3) Prepare Data<br/>3.1) Cleaning<br/>3.2) split out train/test dataset <br/><br/>  4) Evaluate Algorithms<br/>4.1) Algorithms                   |
| PART3 | 5) Improve Accuracy<br/>5.1) Grid Search <br/><br/>  6) Performance of the best algorithms <br/>6.1) check the performance<br/>6.2) futher process |

<br/>
<br/>
<br/>
# PART1: 1) Prepare Problem
<br/>
```python
import pandas as pd
import numpy as np

# 데이터 시각화 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# 데이터 모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 데이터 모델 성능 평가
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss

```
<br/>
```python
# train 데이터 불러오기 
plant_train = pd.read_csv('plant_train.csv')
# test 데이터 불러오기 
plant_test = pd.read_csv('plant_test.csv')
# train 데이터 정보보기 
plant_train.info()
```
<br/>
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 199991 entries, 0 to 199990
Data columns (total 22 columns):
type          199991 non-null object
fiberID       199991 non-null int64
psfMag_u      199991 non-null float64
psfMag_g      199991 non-null float64
psfMag_r      199991 non-null float64
psfMag_i      199991 non-null float64
psfMag_z      199991 non-null float64
fiberMag_u    199991 non-null float64
fiberMag_g    199991 non-null float64
fiberMag_r    199991 non-null float64
fiberMag_i    199991 non-null float64
fiberMag_z    199991 non-null float64
petroMag_u    199991 non-null float64
petroMag_g    199991 non-null float64
petroMag_r    199991 non-null float64
petroMag_i    199991 non-null float64
petroMag_z    199991 non-null float64
modelMag_u    199991 non-null float64
modelMag_g    199991 non-null float64
modelMag_r    199991 non-null float64
modelMag_i    199991 non-null float64
modelMag_z    199991 non-null float64
dtypes: float64(20), int64(1), object(1)
memory usage: 33.6+ MB
```
<br/>
train 데이터를 불러와 구성을 살펴보면 대략 20만개의 데이터가 있고 feature는 type target을 제외한 21개를 포함한 것으로 볼 수 있다. type 변수는 object로 구성되어있고, fiberID 변수는 int로 구성되어 있다는 점을 주의하고 다음 데이터 전처리 단계에서 이 부분을 처리해보도록 하자. 
<br/>
<br/>
# 2) Summarize Data
<br/>
```python
#2.1) Descriptive statistics

plant_train.describe()

#type -> target value
#fiberid -> categorical value 
#psfMag_u의 mean: -6.750146e+00
```
<br/>
<br/>
 Default | fiberID |      psfMag_u |      psfMag_g |      psfMag_r |      psfMag_i |      psfMag_z |    fiberMag_u |     fiberMag_g |    fiberMag_r |    fiberMag_i |  ... |    petroMag_u |    petroMag_g |    petroMag_r |    petroMag_i |    petroMag_z |    modelMag_u |    modelMag_g |    modelMag_r |    modelMag_i | modelMag_z    
 :---- | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | -------------: | ------------: | ------------: | ---: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------- 
 count | 199991.000000 |  1.999910e+05 | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 |  1.999910e+05 |  199991.000000 | 199991.000000 | 199991.000000 |  ... | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 | 199991.000000 
 mean  |    360.830152 | -6.750146e+00 |     18.675373 |     18.401235 |     18.043495 |     17.663526 |  1.084986e+01 |      19.072693 |     19.134483 |     18.183331 |  ... |     21.837903 |     18.454136 |     18.481525 |     17.686617 |     17.699207 |     20.110991 |     18.544375 |     18.181544 |     17.692395 | 17.189281     
 std   |    225.305890 |  1.187678e+04 |    155.423024 |    127.128078 |    116.622194 |    123.735298 |  4.172116e+03 |     749.256162 |     90.049058 |    122.378972 |  ... |    789.472333 |    154.376277 |     97.240448 |    145.730872 |    142.691880 |    122.299062 |    161.728183 |    133.984475 |    131.183416 | 133.685138    
 min   |      1.000000 | -5.310802e+06 | -40022.466071 | -27184.795793 | -26566.310827 | -24878.828280 | -1.864766e+06 | -215882.917191 | -21802.656144 | -20208.516262 |  ... | -24463.431833 | -25958.752324 | -23948.588523 | -40438.184078 | -30070.729379 | -26236.578659 | -36902.402336 | -36439.638493 | -38969.416822 | -26050.710196 
 25%   |    174.000000 |  1.965259e+01 |     18.701180 |     18.048572 |     17.747663 |     17.425523 |  1.994040e+01 |      18.902851 |     18.259352 |     17.903615 |  ... |     19.247795 |     18.113933 |     17.479794 |     17.050294 |     16.804705 |     19.266214 |     18.076120 |     17.423425 |     16.977671 | 16.705774     
 50%   |    349.000000 |  2.087136e+01 |     19.904235 |     19.454492 |     19.043895 |     18.611799 |  2.104910e+01 |      20.069038 |     19.631419 |     19.188763 |  ... |     20.366848 |     19.586559 |     19.182789 |     18.693370 |     18.174592 |     20.406840 |     19.547674 |     19.143156 |     18.641756 | 18.100997     
 75%   |    526.000000 |  2.216043e+01 |     21.150297 |     20.515936 |     20.073528 |     19.883760 |  2.233754e+01 |      21.385830 |     20.773911 |     20.331419 |  ... |     21.797480 |     21.004397 |     20.457491 |     20.019112 |     19.807652 |     21.992898 |     20.962386 |     20.408140 |     19.968846 | 19.819554     
 max   |   1000.000000 |  1.877392e+04 |   3538.984910 |   3048.110913 |   4835.218639 |   9823.740407 |  4.870154e+03 |  248077.513380 |  12084.735440 |   8059.638535 |  ... | 298771.019041 |  12139.815877 |   7003.136546 |   9772.190537 |  17403.789263 |  14488.251976 |  10582.058590 |  12237.951703 |   4062.499371 | 7420.534172   
<br/>
fiberID, psfMag_u의 mean값을 보면 다른 feature들과 조금은 다른 것을 확인할 수 있다. fiberID는 int이기 때문에 편의상 제거를 해보고, psfMag_u는 scaling을 통해 값을 조정할 수 있지만, 다른 feature들의 mean는 대부분 동일하기 때문에 이 변수를 어떻게 처리해야 할지는 좀 더 두고 보겠다. 
<br/>
<br/>
```python
plant_train.type.value_counts()
```
```python
QSO                    49680
GALAXY                 37347
SERENDIPITY_BLUE       21760
SPECTROPHOTO_STD       14630
REDDEN_STD             14618
STAR_RED_DWARF         13750
STAR_BHB               13500
SERENDIPITY_FIRST       7132
ROSAT_D                 6580
STAR_CATY_VAR           6506
SERENDIPITY_DISTANT     4654
STAR_CARBON             3257
SERENDIPITY_RED         2562
STAR_WHITE_DWARF        2160
STAR_SUB_DWARF          1154
STAR_BROWN_DWARF         500
SKY                      127
SERENDIPITY_MANUAL        61
STAR_PN                   13
Name: type, dtype: int64
```
<br/>
type 변수의 값들을 보면 천체의 종류가 나와있고 조금은 unbalance 된 값을 볼 수 있다. 또한 type은 object이기 때문에 이를 문자가 아닌 숫자로 변환하는 작업을 데이터 전처리 과정에서 처리해보자.  
<br/>
<br/>
```python
features = plant_train.drop(['type','fiberID'], axis=1)
features = features.columns.tolist()

corr = plant_train[features].corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(data=corr,annot=True)
plt.show()
```
<br/>
<center><img src="https://github.com/stelch1234/stelch1234.github.io/blob/master/img/heatmap.png?raw=true" style="zoom:60%;"/></center>
<br/>
features 간의 상관관계를 보게되면 특이 한 점을 볼 수 있다. psfMag_u와 fiberMag_u의 상관관계가 1인 점이다. 이전에서 부터 psfMag_u는 값의 범위가 다른 features과 달랐고 상관관계 또한 다른 변수와 높은 것으로 확인됐기 때문에 이 변수는 다음 PART2의 전처리 단계에서 제거해보도록 하겠다. 
<br/>
<br/>
<br/>