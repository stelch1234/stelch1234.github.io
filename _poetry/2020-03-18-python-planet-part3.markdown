---
layout: post
title: 천체 데이터 분류_PART(3)_모델성능
tags: [Python, 분류, 예측]
description: DACON 천체 예측하기
comments: true
---
<br/>

# PART3: 5) Improve Accuracy
<br/>

```python
#5.1) Grid Search

# 1단계 grid search
n_estimators = [100,300,500,700]
max_depth = [15,30,40]

rf_param_grid = dict(n_estimators = n_estimators,max_depth = max_depth)
rf = RandomForestClassifier(random_state = 777)
rf_cv = GridSearchCV(estimator = rf, 
                     param_grid = rf_param_grid,
                     cv = 5,
                     verbose = 2, 
                     n_jobs = -1)
rf_grid_result = rf_cv.fit(X_train, y_train)

# 1단계 grid search 결과 정리
print("Best: %f using %s" % (rf_grid_result.best_score_, rf_grid_result.best_params_))
means = rf_grid_result.cv_results_['mean_test_score']
stds = rf_grid_result.cv_results_['std_test_score']
params = rf_grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

이전 part2에서 randomfroest 모델평가가 가장 좋았기 때문에 랜덤 포레스트 모델에 grid search해 파라미터 튜닝을 해보려고 한다. 
한번에 많은 파리미터를 튜닝해서 grid search를 진행하기 보다는 순차적으로 단계를 나눠서 grid search의 결과를 보고자 한다. 

첫번째 grid search에서는 n_estimators = 700, max_depth = 30일때가 가장 좋은 결과가 나왔다. 
<br/>
<br/>

```python
# 2단계 grid search
min_samples_leaf = [2,4,6,8]
min_samples_split = [5,10,20]
criterion = ['gini', 'entropy']

param_grid = dict(min_samples_leaf = min_samples_leaf, 
                  min_samples_split = min_samples_split,
                  criterion = criterion)

rf2 = rf_grid_result.best_estimator_
rf_cv2 = GridSearchCV(estimator = rf2,
                   param_grid = param_grid,
                   cv = 3,
                   verbose = 2, 
                   n_jobs = -1)
rf_grid_result2 = rf_cv2.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (rf_grid_result2.best_score_, rf_grid_result2.best_params_))
means = rf_grid_result2.cv_results_['mean_test_score']
stds = rf_grid_result2.cv_results_['std_test_score']
params = rf_grid_result2.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

```python
Best: 0.865865 using {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 5}
0.865865 (0.000679) with: {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 5}
0.864708 (0.000453) with: {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 10}
0.862829 (0.000872) with: {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 20}
..
0.861322 (0.000655) with: {'criterion': 'entropy', 'min_samples_leaf': 8, 'min_samples_split': 10}
0.860650 (0.000631) with: {'criterion': 'entropy', 'min_samples_leaf': 8, 'min_samples_split': 20}
```
첫번째 튜닝의 값이 제일 좋았던 모델을 바탕으로 두번째 grid search를 진행한다.  파라미터는 leaf, split, criterion를 조정해주었다. 
그 결과 첫번째 성능의 결과보다 두번째 grid search를 했을때가 더 성능이 좋지 않게 나왔다. 해당 파라미터는 적용하지 않는 것으로 하자. 
<br/>
<br/>

# 6) Performance of the best algorithms

```python
# 6.1) check the performance
random_df2 = RandomForestClassifier(n_estimators = 700, max_depth = 30,
                                   random_state = 777)
random_df2.fit(X_train,y_train)
random_df_pred2 = random_df2.predict_proba(X_test)
random_df_pred_log2 = log_loss(y_test, random_df_pred2)
print('grid search 한 randomforest logloss 값:', random_df_pred_log2)
```
```python
grid search 한 randomforest logloss 값: 0.40617618269231676
```

grid search를 통해 tuning한 파라미터를 넣고 다시 모델을 만들어 성능 평가를 했다. 그 결과 logloss 값이 0.41인 것으로 나왔다. 
이는 앞서 단순 randomforest 모델의 결과 1.3인거에 비해 현저하게 logloss값이 줄어들어 모델 성능이 좋아진 것을 볼 수 있다.
<br/>
<br/>

```python
# 6.2) futher process

# 변수 중요도 구하기 
importances = random_df.feature_importances_
x_features = X_train.columns.tolist()

# 변수, 중요도를 list형태로 반환
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x_features, importances)]

# list를 변수중요도에 따라 정렬
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('변수: {:20} 중요도의 정도: {}'.format(*pair)) for pair in feature_importances]

#psfMag_z , modelMag_i가 가장 중요한 변수
```

```python
변수: fiberMag_u           중요도의 정도: 0.08
변수: modelMag_z           중요도의 정도: 0.08
변수: psfMag_g             중요도의 정도: 0.07
변수: psfMag_z             중요도의 정도: 0.06
변수: fiberMag_g           중요도의 정도: 0.06
변수: petroMag_z           중요도의 정도: 0.06
변수: modelMag_g           중요도의 정도: 0.06
변수: modelMag_r           중요도의 정도: 0.06
변수: modelMag_i           중요도의 정도: 0.06
변수: psfMag_r             중요도의 정도: 0.05
변수: psfMag_i             중요도의 정도: 0.05
변수: fiberMag_z           중요도의 정도: 0.05
변수: modelMag_u           중요도의 정도: 0.05
변수: petroMag_u           중요도의 정도: 0.04
변수: petroMag_g           중요도의 정도: 0.04
변수: petroMag_r           중요도의 정도: 0.04
변수: petroMag_i           중요도의 정도: 0.04
변수: fiberMag_r           중요도의 정도: 0.03
변수: fiberMag_i           중요도의 정도: 0.03
```

좀더 성능을 높여보기 위해 불필요한 변수가 있는지 살펴보고 변수 제거해보고자 했다. 랜덤포레스트 모델의 feautre importance를 통해 변수 중요도를 살펴보자.

list를 정렬할때는 sorted(데이터 , key=lambda x : 기준, reverse = True)을 활용하며, reverse = True는 내림차순 정렬을 뜻한다.  
<br/>
<br/>

```python
# 변수 중요도 plt 그리기
x_values = range(len(importances))
plt.figure(figsize=(20,10))
plt.bar(x_features, importances, color = 'r', edgecolor = 'k', linewidth = 1.2)
plt.xticks(x_values, x_features, rotation='vertical')
plt.ylabel('Importance') 
plt.xlabel('Variable') 
plt.title('Variable Importances')
```
<center><img src="https://github.com/stelch1234/stelch1234.github.io/blob/master/img/bar.png?raw=true"/></center>

각 변수의 중요도를 bar 그래프로 나타냈다.
<br/>
<br/>

```python
# feautre list & importance list구하기 
sorted_featrues = [importance[0] for importance in feature_importances]
sorted_importances = [importance[1] for importance in feature_importances]

# importance list의 누적합  
cumulative_importance = np.cumsum(sorted_importances)

# plt 그리기
plt.figure(figsize=(20,10))
plt.plot(x_values, cumulative_importance, 'b')

# 경계선 설정하기 (95%)
plt.hlines(y=0.95, xmin=0, xmax=len(sorted_importances), color='r', linestyles='dashed')
plt.xticks(x_values, sorted_featrues, rotation='vertical')

#axis label and title
plt.xlabel('Variable')
plt.ylabel('Cumulative importance')
plt.title('Cumulative importances')
```

<center><img src="https://github.com/stelch1234/stelch1234.github.io/blob/master/img/graph.jpg?raw=true"/></center>

각 feture importance의 누적값 경계를 95로 설정하여 이를 그래프화 시켰다.
<br/>
<br/>

```python
#구체적으로 몇번째에서 threshold를 넘는지 확인 
print('중요도 총합이 95%가 되는 feature의 갯수 :', 
      np.where(cumulative_importance > 0.95 )[0][0] +1)
```

```python
중요도 총합이 95%가 되는 feature의 갯수 : 17
```

설명력이 95가 되는 시점의 feature 갯수는 17이다. 
<br/>
<br/>

```python
# 중요도 총합이 95%가 되는 상위 변수들만 추출 
important_feature_names = [feature[0] for feature in feature_importances[0:16]]

important_train_features = X_train[important_feature_names]
important_test_features = X_test[important_feature_names]
print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)
```
```python
Important train features shape: (139993, 16)
Important test features shape: (59998, 16)
```

중요 변수17개만 들어가는 train & test 데이터셋을 만들었다.
<br/>
<br/>

```python
# training and evaluating on important features 
random_df2.fit(important_train_features,y_train)
random_df_pred3 = random_df2.predict_proba(important_test_features)
random_df_pred_log3 = log_loss(y_test, random_df_pred3)
print('importanct feautre만 포함한 randomforest logloss 값:', random_df_pred_log3)
```
```python
importanct feautre만 포함한 randomforest logloss 값: 0.4291484080235962
```

안타깝게도.. 변수가 제거된 데이터 셋에 랜덤 모델에 적용해 그 결과값을 확인하니 logloss값이 다소 늘어났다.
logloss는 늘었지만, 변수가 줄어들게 되면 모델 성능 속도가 이전 보다 빨라지니 나중에 다른 더 많은 변수를 지니고 있는 데이터를 다룰때 고려해보면 좋은 방법이다.
<br/>
<br/>

# 7) Fianlize Model
```python
#7.1) create fianal model
# plant_test 데이터 정제 
plant_test = plant_test.drop(['psfMag_u','fiberID'], axis=1)

# 최종 모델 훈련 
random_df2.fit(X,y)

#7.2) predictions on test datase
# 최종 모델로 예측 데이터 생성
y_pred = random_df2.predict_proba(plant_test)
sample_submission = pd.read_csv('sample_submission.csv').set_index('id')

# submission 파일 생성 
submission = pd.DataFrame(data = y_pred, columns = sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)
```

결국은 중요 feature를 제거하지 않은 데이터 셋을 가지고 최종적으로 모델을 훈련하여 test의 y값을 예측했다.

<br/>
<br/>
<br/>

