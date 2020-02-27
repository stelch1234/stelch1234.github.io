---
layout: post
title: 데이터 스케일링 (Data Scaling)
tags: [Python, 스케일링]
description: 데이터 스케일링 방법
comments: true
---

특정 feature의 값이 너무 크고, 또다른 feature의 값은 너무 작을때 모델 알고리즘 학습과정에서 0으로 수렴해버리거나 무한대로 발산하게 된다. 

이러한 이유로 모델링을 진행하기 전에 데이터 스케일링(Scaling) 을 해주는 과정이 필요하며 대부분의 스케일링들은 이상치(outlier)에 민감하기 때문에 이상치 처리가 필수적으로 해야한다.특히 데이터 스케일링은 k-means 와 같은 거리 기반의 모델에서 중요하게 된다. 

스케일링의 또 다른 장점은 , 스케일링을 해주면 다차원의 값들을 비교 분석하기 쉽게 되고 오버플로우(overflow)와 언더플로우(underflow)를 방지있다.  
<br/>
<br/>
# 스케일링 (Scaling) 종류
------

# StandardScaler 
<br/>
기본 스케일로, 각 feature의 값들의 평균은 0, 분산은 1로 변경해준다. 이상치에 민감하다는 단점이 있다.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train  = scaler.fit_transform(X_train)
#fit는 데이터 변환을 학습,
#transform은 실제 데이터 스케일 조정,
#fit과 transform을 동시에 진행하는게 fit_transform().
```
<br/>

# RobustScaler 
<br/>
모든 특성들이 같은 크기를 갖는다는 점에서 StandardScaler와 비슷하지만, 평균과 분산 대신 median과 quartile를 사용한다. 이상치 영향을 받지 않는다는 장점이 있다.

```python
from sklearn.preprocessing import StandardScaler
scaler = RobustSacler()
X_train  = scaler.fit_transform(X_train)
```
<br/>

# MinmaxScaler
<br/>
모든 feature의 값들이 0~1사이에 위치하게 되어 feature의 최대 최소 값이 각각 1,0이 된다. 이상치에 민감하다는 단점이 있다.

```python
from sklearn.preprocessing import StandardScaler
scaler = MinMaxScaler()
X_train  = scaler.fit_transform(X_train)
```
<br/>

# Normalizer
<br/>
위의 scaler는 각 컬럼의 통계치를 이용해 데이터의 값을 변환을 주지만, Normalizer는 각 row마다 정규화해 유클리드 거리가 1이 되도록 데이터 조정한다.

```python
from sklearn.preprocessing import StandardScaler
scaler = Normailizer()
X_train  = scaler.fit_transform(X_train)
```

