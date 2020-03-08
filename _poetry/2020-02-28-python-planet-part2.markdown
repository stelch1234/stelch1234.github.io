---
layout: post
title: 천체 데이터 분류_PART(2)_전처리&이상치제거
tags: [Python, 분류, 예측]
description: DACON 천체 예측하기
comments: true
---
<br/>

## PART2: 3) Prepare Data

<br/>

```python
# 3.1) Cleaning
# 이상치 검출 
def outlier_detect(df):
    outlier_indices = []
    # iterate over features(columns)
    for feature in features:
        # 1st quartile (25%)
        Q1 = np.percentile(plant_train[feature], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(plant_train[feature], 75)
        # Interquartile rrange (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        
        # outlier가 존재하는 관측치 추출 
        outlier_list_feature = plant_train[(plant_train[feature] < Q1 - outlier_step) | 
                                       (plant_train[feature] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_feature) #extend는 list 끝에 iterable 모든 항목 넣음
        
    # 이상치가 존재하는 관측치 검출
    outlier_indices = Counter(outlier_indices)
    # itmes는 key, value로 묶어서 list return 
    multiple_outliers = [k for k, v in outlier_indices.items() if v > 5]  
    return multiple_outliers   

print('데이터 셋에서 5개이상의 이상치가 존재하는 관측치는 %d개이다.' %(len(outlier_detect(plant_train[features])))) 
```

```python
데이터 셋에서 5개이상의 이상치가 존재하는 관측치는 356개이다.
```

머신러닝 알고리즘 중에서 이상치에 민감한 알고리즘들이 있기 때문에 데이터 전처리 과정 중에서 이상치 탐색은 중요한 과정이라고 할 수 있다.
보통 이상치 탐색은 boxplot 시각화를 통해 확인하고 검출해 낼 수 있지만, feature가 많고 데이터 수가 많은 데이터 셋에서는 하나하나 살펴보기가 어렵다. 이를 위해 여기서는 percentile을 활용해 갹갹의 feature 내에 이상치를 갖는 관측치들을 추출하였다. 그 결과 5개 이상의 feature에서 이상치가 존재하는 관측치는 356개가 있는 것으로 나타났다. 21개의 features 가운데 5개 features 이상에서 이상치가 존재하는 관측치는 제거했다. 

코드를 활용함에 있어 주의할 점은 outlier_indices의 list에서 .append()가 아닌 .extend()를 써야하는 점이다. append()와 extend()의 차이점을 보면 다음 아래와 같다.   

```python
x = [1,2,3]
x.append([1,5])
print(x)
[1,2,3[1,5]] #객체 끝 부분에 추가 
```
```python
x = [1,2,3]
x.extend([1,5])
print(x)
[1,2,3,1,5] #같은 리스트 내에 추가 
```

extend()를 활용함으로써, 각각의 features들이 for문을 통해 돌때마다 이상치가 존재하는 관측치를 하나의 리스트에다가 계속 추가한다. 그 다음 Counter()를 활용해서 이상치가 존재하는 관측치로 총 몇번 나왔는지 알게된다. 
<br/>
<br/>

```python
# 이상치 검출된 관측치 제거
outlier_indices = outlier_detect(plant_train[features]) 
plant_train = plant_train.drop(outlier_indices).reset_index(drop=True)
print(plant_train.shape)
```

```python
(199635, 22)
```

5개의 features에서 이상치가 존재하는 356개의 관측치를 제거하면 총 199,635 관측치가 남게 되는 것을 확인할 수 있다. 

<br/>
<br/>

```python
# psfMag_u 평균값과 상관분석의 결과로 보아 해당 칼럼 삭제
plant_train = plant_train.drop(['psfMag_u','fiberID'], axis=1)

# type변수 type을 object -> int변환 
encoder = LabelEncoder()
plant_train['type'] = encoder.fit_transform(plant_train['type'])
plant_train.type.value_counts()
```

```python
1     49680
0     37347
4     21760
10    14630
2     14618
16    13750
11    13500
6      7132
3      6580
14     6506
5      4654
13     3257
8      2562
18     2160
17     1154
12      500
9       127
7        61
15       13
Name: type, dtype: int64
```

앞서 part1에서 psfMag_u는 전체 평균값이 다른 features과 상이하게 다르고 상관관계에서 다른 feature와 1인 아주 높은 상관성을 보였기 때문에 계산의 편의상 해당 칼럼을 삭제 했다. 

type변수가 object이기 떄문에 추후 모델링을 위해 이를 int로 변환해주었다. 


```python
# 3.2) split out train/test dataset

X = plant_train.drop(['type'], axis=1)
y = plant_train['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
```

```python
Training Features Shape: (139993, 19)
Training Labels Shape: (139993,)
Testing Features Shape: (59998, 19)
Testing Labels Shape: (59998,)
```
<br/>
<br/>

## 4) Evaluate Algorithms

```python
#4.1) Algorithms

# randomforest
random_df = RandomForestClassifier(class_weight='balanced',random_state=777)
random_df.fit(X_train,y_train)
random_df_pred = random_df.predict_proba(X_test)
random_df_pred_log = log_loss(y_test, random_df_pred)
print('기본 randomforest logloss 값:', random_df_pred_log)
# 기본 randomforest logloss 값: 1.3466678834352566
```

```python
기본 randomforest logloss 값: 1.3466678834352566
```

기본설정으로 랜덤포레스트를 활용해 classificaiton한 결과를 보면 logloss값이 1.3인 것으로 나타났다. logloss는 낮을 수록 좋다. 

```python
# Bagging+DecisionTree
from sklearn.ensemble import BaggingClassifier
bag_decision_df = BaggingClassifier(DecisionTreeClassifier())
bag_decision_df.fit(X_train, y_train)
bag_df_pred = bag_decision_df.predict_proba(X_test)
bag_df_pred_log = log_loss(y_test, bag_df_pred)
print('Bagging logloss 값:', bag_df_pred_log)
# 기본 Bagging_DecisionTree log loss 값: 1.4715524355664136
```

```python
Bagging logloss 값: 1.4715524355664136
```

배깅을 활용해 classificaiton한 결과를 보면 logloss값이 1.47인 것으로 나타났다. 랜덤포레스트보다 logloss 값이 좋지 않기때문에 이 이후에는 grid search를 활용해 랜덤포레스트의 파라미터를 설정해보고자 한다. 

<br/>
<br/>
<br/>
