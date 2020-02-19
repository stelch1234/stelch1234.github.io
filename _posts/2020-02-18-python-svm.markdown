---
layout: post
title: "서포트 벡터 머신 (SVM)"
categories: python
author: 'Stella'
tags: [Python, SVM]
#image: assets/images/1.jpg
comments: true
---



서포트 벡터 머신 (SVM)은 이진분류로 로지스틱 회귀나 결정트리와 같은 분류 모델 보다 더 좋은 정확성을 나타낸다. 선형 또는 비선형 분류뿐만 아니라 회귀, 이상치 탐색에도 사용할 수 있는 모델이며 특히 복잡한 분류 문제에 잘 맞는다. 그래서 얼굴 인식, 이메일 분류, 기사 분류, 유전자 분류, 손글씨 인식과 같이 다양하게 사용이 되고 있다. 

하지만 의사결정나무처럼 직관적인 해석이 불가능하다는 점이 최대 단점이다. 어떻게 데이터들이 분류됐는지 알 수가 없기 때문에 결과 해석에 있어서는 의사결정나무가 사용되고 있지만 높은 정확도를 위해서는 서포트벡터머신 (SVM)을 사용한다.

<br/>
**장점**: 강력한 모델이며 다양한 데이터셋에 잘 작동, 차원 수 > 데이터 수 일떄 효과적
<br/>

**단점**: 10만개 이상의 데이터셋에서는 잘 맞지 않음, 전처리 스케일링 필수, 모형 구축에 많은 시간 필요, 해석의 어려움


<br/>
## 서포트벡터머신 (SVM) 이란?

[![img](https://postfiles.pstatic.net/MjAyMDAxMTdfMTk4/MDAxNTc5MjY5MDQ0ODUz.YUsyX1iUxGiIQBr_y4sZcOKhXc-iWJJIs-iQ9_FlOnog.jWHFP2i9pcZUP1WJJxRPzreaInm6EsJG9-aNO6_7EKcg.PNG.stelch/svm.png?type=w966)](https://blog.naver.com/PostList.nhn?blogId=stelch&widgetTypeCall=true&from=section&topReferer=https%3A%2F%2Fsection.blog.naver.com%2FBlogHome.nhn%3FdirectoryNo%3D0%26currentPage%3D1%26groupId%3D0&directAccess=true#)

서포트벡터머신(SVM)의 기본적인 아이디어는 위의 그림 처럼 별과, 세모 사이의 여백(margin)을 최대한 넓게 둘 수 있는 결정 초평면(decision hyperplane)을 찾아 데이터를 잘 분류하는 것이다. 



여기서 초평면을 직선으로 나눌 수 있으면 ---> 선형 분류 모델

직선으로 나눌 수 없으면 ---> 비선형 분류 모델 


<br/>
## 서포터벡터머신 (SVM)을 이해하기 위한 용어설명 

#### 초평면 (Hyperplane)

최대 마진 분류기 (Maximal Margin Classifier)가 선형 경계로 사용하는 선을 초평면 (Hyperplane)이라 부르고, 데이터가 n차원이면 초평면은 n-1차원을 가진다.

#### 서포트 벡터 (Support Vector)

초평면(Hyperplane)까지의 거리가 가장 짧은 데이터 벡터를 서포트 벡터(support vector)라고 부른다. 

서포트 벡터로 인해 서포트벡터머신 (SVM)이 가지는 장점은 새로운 데이터가 들어왔을 때 전체 데이터와의 내적 거리를 보지 않고 서포트 벡터와의 내적 거리만 구하기만 되기 때문에 cost 값을 줄일 수 있다.


<br/>
## 비선형 분류를 위한 커널트릭 (Kernal Trick)

[![img](https://postfiles.pstatic.net/MjAyMDAxMTdfMjUz/MDAxNTc5MjcwMDMxMTE4.AgDYdiiw71rMidWM_r6HpY_HE45JXqWkEbwfKY9WF9og.jqCDovTbqlsrpH0aGn9MCJmeaajngtDORE557I6p9eEg.PNG.stelch/svm_kernal_trick.png?type=w966)](https://blog.naver.com/PostList.nhn?blogId=stelch&widgetTypeCall=true&from=section&topReferer=https%3A%2F%2Fsection.blog.naver.com%2FBlogHome.nhn%3FdirectoryNo%3D0%26currentPage%3D1%26groupId%3D0&directAccess=true#)

서포트벡터머신 (SVM)은 선형분류일때는 선으로 분류할 수 있지만, 비선형 분류에서는 주어진 데이터를 고차원으로 인식해 분류해야 하기 때문에 커널 트릭(Kernel Trick)을 사용해야 한다. 커널(Kernal)을 사용하면 복잡한 데이터에서도 차원을 간단하게 바꿀 수 있게 된다. 이렇게 함으로써 선형으로 해결할 수 없는 문제들을 해결하게 된다. 


<br/>
### 서포트벡터머신 (SVM) 활용 코드

```python

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm = SVC(kernel='rbf', C=10,gamma=0).fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print('SVM의 정확도:', accuracy_score(y_test, X_test) )

```
- gamma : 1~0사이의 값으로 데이터의 범위이다. 값이 적을수록 결정 경계에 대한 데이터 포인트의 영향 범위가 커진다.
<br/>
- C: 규제 매개변수. 값이 적을수록 매우 제약이 큰 모델을 만들어 선형에 가까우며 잘못 분류된 데이터 포인터 경계에 거의 영향을 주지 않는다. 값이 증가하면 직선이 아닌 굴곡진 경계를 만들어 데이터들을 정확히 분류하게 한다. 

