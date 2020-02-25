---
layout: post
title: "로지스틱회귀(logistic regression)"
categories: python
author: 'Stella'
tags: [Python, 로지스틱]
#image: assets/images/1.jpg
comments: true
---



## 로지스틱 회귀 (Logistic Regression) 란?

로지스틱 회귀(logistic regression)는 주로 특정 분류를 위해 자주 사용되는 알고리즘으로 참과 거짓, 합격과 불합격과 같이 이진분류를 할 때 사용될 수 있다.

<br/>
<br/>

## 시그모이드 (Sigmoid)

![img](/Users/stella/Downloads/stelch1234.github.io/_assets/images/sigmoid.png){: with="100"}

로지스틱 회귀는 0~1사이의 값을 가져야 한다. 일반적으로 우리가 알고 있는 회귀식 h(x) = wx+b 에서는 0~1사이의 값을 가지기가 어렵다. 그렇기 때문에 0~1사이의 값으로 변환하기 위해 필요한 것이 있는데, 그것이 바로 시그모이드(sigmoid)이다.

시그모이드 함수의 수식은 다음과 같다.

![img](/Users/stella/Downloads/stelch1234.github.io/_assets/images/4564462177353728.png)

이 함수를 통해 우리가 원하는 0~1사이의 값으로 예측할 수 있게 된다. x의 값이 증가하게 되면 1로 수렴하게 되고 x의 값이 감소하면 0으로 수렴하게 된다. 식을 자세히 설명하면, 위 식에서 e는 자연 상수를 말하고 a와 b는 다음 아래를 보면 이해할 수 있다.


![img](/Users/stella/Downloads/stelch1234.github.io/_assets/images/sigmoid_a.png)


빨간색은 a=0.5일때, 초록색은 a=1일때, 파란색은 a=2일 때이다. a의 값에 따라 그래프 경사가 변화되고 있는 것을 볼 수 있다. 이를 통해 여기서 a의 값은 그래프 경사도라는 걸 알 수 있다.


![img](/Users/stella/Downloads/stelch1234.github.io/_assets/images/sigmoid_b.png)

빨간색은 b의 값이 클때이고, 파란색은 b의 값이 작을때를 의미한다. 즉, b값에 따라 그래프가 이동하는 것을 알 수 있다. 

<br/>
<br/>
## 비용함수 (Cost Function)

시그모이드를 통해 우리가 원하는 0~1사이의 값을 만들었으면 이제는 이것이 실제 값과 얼마나 차이가 있는지를 알아봐야 한다. 이를 나타내기 위해서는 비용함수 (Cost Function)에 대한 정의가 필요하다. 비용함수 (Cost Function)는 예측값과 실제 값 차이의 평균을 나타내는 함수이다. 보통 비용함수 (Cost Function)를 나타내는 그래프는 매끈한  U형의 그래프이다. 이 매끈한 골짜기 모양에 경사 하강법(Gradient descent)를 사용해 Cost가 최소가 되는 지점을 찾게 된다.  

우리가 알고 있는 이 비용함수에 로지스틱을 적용하게 되면 우리가 원하는 U형의 그래프가 나타나게 될까? 

![img](/Users/stella/Downloads/stelch1234.github.io/_assets/images/Non_convex.png){: with="80"}

안타깝게도, 우리가 원하는 U형의 매끈한 그래프가 만들어지기 어렵다. 왜냐하면 로지스틱 회귀식에서 본 것처럼 로지스틱 회귀식에는 자연상수인 e가 들어가 있는다. 이 자연상수로 인해서 그래프가 울퉁불퉁한 안 예쁜 그래프 모양을 지니게 된 것이다. 이 그래프 상태에서 경사하강법을 적용하게 되면 비용함수의 최소값(global minima)에 수렴하지 않고 중간에 작은 골짜기(local minima)에 수렴하게 된다.


<br/>
매끈한 U형 그래프를 만들어 global minima에 도달하기 위해서 로그 함수를 활용하면 된다.

![img](/Users/stella/Downloads/stelch1234.github.io/_assets/images/log.png){: with="50"}

y=0.5에 대칭하는 두개의 로그 함수를 그리면 위와 같다. 

파란색은 실제 값이 1일때, 사용되는 그래프로 0에 가까울수록 오차가 커져 Cost가 커지는 것을 볼 수 있다.
빨간색은 실제 값이 0일때 ,사용되는 그래프로 0에 가까워야 오차가 작아 Cost도 적어지는 것으로 확인될 수 있다.

파란색과 빨간색의 식은 각각 -log h 와 -log(1-h)로 나타나게 된다. 실제 값이 1일 때 파란색 식 -log h를 써야 하고 실제 값이 0일 때 빨간색 식  -log(1-h)을 써야 한다. 로그함수를 이용해 우리가 원하는 최종적인 비용 함수를 다음 수식처럼 정의할 수 있게 된다. 

![img](/Users/stella/Downloads/stelch1234.github.io/_assets/images/5961578112090112.png)

