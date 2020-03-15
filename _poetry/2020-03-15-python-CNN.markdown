---
layout: post
title: 컨볼루션 신경망 (CNN)
tags: [Python, CNN, 컨볼루션]
description: 컨볼루션 신경망
comments: true
---

# 컨볼루션 신경망(CNN) 이란? 
----
컨볼루션 신경망은 딥러닝 알고리즘 중에서 가장 많이 사용되는 알고리즘이다. 주로 사용은 이미지, 비디오, 텍스트, 사운드, 얼굴 인식 등.. 다양한 영역에서 특징 추출 또는 분류를 위해 사용되고 있다. 

<br/>
**컨볼루션의 장점**
- 패턴을 직접 찾고 특징을 분류하는데 직접 학습하기 때문에 수동 작업이 필요하지 않다.
- 높은 수준의 인식 결과를 나타낸다. 

<br/>
<br/>

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMTMg/MDAxNTUzNTMxNDg5Nzc2.NgsUoczUl4mPC8vdLHSiS_F4mr3rECDChiONEQ6X5RAg.nxu2AjdlV6-sVQ7xpbYUswGgks5JKS0W3HsmyXvZS1Eg.JPEG.stelch/%EC%BB%A8%EB%B3%BC%EB%A3%A8%EC%85%98_%EC%84%A4%EB%AA%85.jpeg?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)


위에서 볼 수 있는 것처럼 각각의 입력 데이터가 필터(커널)을 통해 컨볼루션 층을 통과하게 되면서 특징을 추출하고 최종적으로 Softmax 함수를 통해 특징을 0~1사이의 값으로 분류하는 과정을 거치게 된다. 
<br/>
<br/>
# 컨볼루션 신경망 (CNN) 용어 설명
----
**Convolution(합성곱)** 

[![img](https://postfiles.pstatic.net/MjAxOTAzMjVfNDUg/MDAxNTUzNTAwMjkwNzgz.FqV4lLP3zXSVWpdMXu2U5eFcLFdnN_NhSdFqf9NzZnsg.sXJAFgFIzR5KS1TAHLLSSMZEjhghYmqfpoDFsoZAYKcg.GIF.stelch/Convolution_schematic.gif?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

아래 그림에서 보는 것처럼 5x5로 이뤄진 입력 데이터에 3x3의 필터를 통과하게 되면서 합성곱 연산이 수행된다. 필터가 지나가면서 나타난 결과값을 Convolved Feather 즉, Feature Map으로 보여주고 있다. 이 과정을 통해 정교한 특징 추출할 수 있게 된다. 케라스에서 컨볼루션 층을 추가하는 함수는 Conv2D()이다. 

```python
model.add(Conv2D(첫번째 인자, kernel_size=( , ), input_shape=( , , ), activation='relu')
#첫번째인자 : 몇개의 필터가 지나갈지 
#kernel_size : 필터의 크기를 말하며 kenrl_size(행, 열)
#input_shape : 입력 값을 말하며 input_shape=(행, 열, 색상 또는 흑백)
#activation : 활성화 함수 
```

<br/>

**Filter(필터)** 

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMzUg/MDAxNTUzNTMxNjUxMTg4.WWwKS2G478QsLC5r9YZV16iFRrlYfxmVbdF8PDW6g-Ag.oAa6QVlkAckJdkhMUXhgTK9-xGbBIcONjJBEtDKgEcEg.PNG.stelch/%EC%BB%A8%EB%B3%BC%EB%A3%A8%EC%85%98_%EC%84%A4%EB%AA%85_.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

필터는 주로 4x4 또는 3x3 행렬로 정의되며 특징 추출을 위해 사용되는 파라미터이다. 지정된 간격으로 이동하면서 전체 입력 데이터와 합성곱을 통해 결과적으로 Feature Map을 만들어 낸다.  어떤 필터 값을 갖는 냐에 따라 나타내는 결과값 또한 달라질 수 있다. 

<br/>

**Stride**

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMTk3/MDAxNTUzNTYwMDA2MzUw.r28tqa60WmNYvvgoIWzU5hMeYo9tClTOogLXzH1FWrEg.l-L4MAMWml8t-JAEkgHYwEKq43A6w08LLrKFIpoC9rcg.PNG.stelch/stride.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

지정된 가격으로 필터가 순회하는 것을 말한다. 아래 그림에서는 stride 값이 2이기 때문에, 2칸씩 이동하는 걸 볼 수 있다. 


<br/>

**Channel(채널)** 

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMTE3/MDAxNTUzNTYxMDU3OTk3.O8vEBEAUN8AqrFZ017PFhW60cfQ0JAJ4uUg0RDQx-Sog.CJo7L-SZfKmxF3vM_gDLn4W3_QNEIcf9FEGK6DfIspAg.PNG.stelch/convolution-operation-on-volume5-2.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

컨볼루션 층에 n개의 필터가 적용되면 출력 데이터는 n개의 채널을 갖는다. 여러 개의 채널이 있을 경우에는 채널을 각각 순회하면서 합성곱하여 채널별 Feature Map을 만든다. 그리고 최종적으로 모든 Feature Map을 합산하여 최종 Feature Map을 만들고 이를 Activation Map이라고 부른다. 

<br/>

**Padding(패딩)**

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMjUx/MDAxNTUzNTYwODk4ODM1.HaIX4k-Ku7TSEnIKMgRWXI0lwYb94NjDsTYYwRydxJgg.Z9WJKUaiUliv-KrbaoUbiC070jGH6sAFOxNaTI6D-LUg.PNG.stelch/padding.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

원래 입력 데이터에서 필터와 Stride 1값으로 적용한 후에는 Feature Map이 작아진다. 여러 단계에 걸쳐 필터를 연속적으로 적용하여 특징을 추출하게 되면 처음에 비해 특징이 소실될 수 있다. 이를 방지하는 것이 패딩이다. 패딩은 입력 데이터의 가장자리 부분에 지정된 픽셀만큼 특정 값으로 0을 채워 넣는다.

<br/>

**Activation Function(활성화 함수)** 

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMjEx/MDAxNTUzNTYxNTg3MjAw.qsgudF0HrV1Yb7RZ4dUIEXDBUKBMIqQl1ljLUCtHD_sg.MWbwVrI1Z4xzpmsMWvQLZ5ayb2ohru1w-Jl-tmI5pWYg.PNG.stelch/1XxxiA0jJvPrHEJHD4z893g.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

필터를 통해 Feature map이 추출되면, 이를 활성화 함수에 적용한다. 이를 통해 입력 데이터의 특징을 정량적으로 분류할 수 있다. 주로 시그모이드(Sigmoid)와 렐루(ReLU)함수 중 렐루가 많이 사용된다. 역전파가 일어날 때 시그모이드(Sigmoid)는 Gradient Vanishing 문제가 발생하기 때문이다. 

<br/>

**Max pooling(맥스 풀링)**

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMTU2/MDAxNTUzNTYyMjc2NDkz.p4NfQIPNzQJVqdbzxrToZ2LzQrW71WyxkEIU_mlOiqwg.8KAW1zDPpM68m5n-oAB0Txe6iylmk-RTIXkaTS099Q0g.PNG.stelch/MaxpoolSample2.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

컨볼루션을 통해 이미지 특징을 추출하지만 여전히 그 특징 추출이 어렵다면 다시 한번 축소 작업을 해야 합니다. 이 과정을 풀링(pooling) 또는 서브 샘플링(sub sampling)라고 한다. 풀링 기법 중 가장 많이 사용되는 기법은 맥스 풀링으로  정해진 구역에서 가장 큰 값만 다음 층으로 넘기는 것을 말한다.  

```python
model.add(MaxPooling2D(pool_size= 2)) 
#pool_size : 풀링 창의 크기를 말하며 2는 전체 크기가 절반으로 줄어든다.
```


<br/>

**Fully Connected Layers(=FC)** 

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMTUw/MDAxNTUzNTU4ODI5NjMy.22eR1d1RCw8YX9DrdBJUgczd0OPzlk8Yg70K0k2JUoQg.GtIZPxSMKn4P5112N7o2oeWzcKsuGoMW7ng6rWwfVXAg.PNG.stelch/The-proposed-ConvNet-approach-uses-of-five-convolutional-layers-with-max-poo.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

첫 번째 FC는 마지막 컨볼루션 층과 연결되어 있다. 그리고 마지막 FC는 바로 이전의 FC와 연결되어 있다. 컨볼루션 층에서는 특징 추출을 했다면, FC는 이 추출된 특징을 기존 신경망에 적용해 분류하는 과정을 거치는 것을 말한다. 

<br/>

**Dropout(드롭아웃)**

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfOTcg/MDAxNTUzNTYxNzI1OTI1.wKf0_hHI_1RpbdBA3FaBYlJUMP0I4eahEWYiLE4czgMg.gSF-hwoiF4WlcFpjo_ET5ImkfWKNtoyG4OaRojRvHqgg.PNG.stelch/9-An-illustration-of-the-dropout-mechanism-within-the-proposed-CNN-a-Shows-a.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

랜덤으로 확률에 따라 드롭아웃하여 의도적으로 학습을 방해한다. 이렇게 함으로써 과적합 현상을 방지하게 된다.

```python
model.add(Dropout(0.25)) 
#Dropout : 몇 %의 노드를 끄려지는지 
```

<br/>

**Softmax(소프트 맥스)** 

[![img](https://postfiles.pstatic.net/MjAxOTAzMjZfMjc0/MDAxNTUzNTYxODM4OTE1.3wSgsey_Lmy3McHUihK0zAw_cx7YiECANmYOsT2cKpEg.aJ3bN6rhIY6esih2xptLZVirTczRdOxZvcTaZu2JdwEg.PNG.stelch/76_blog_image_1.png?type=w966)](https://blog.naver.com/PostView.nhn?blogId=stelch&logNo=221497552593&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=&from=section&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=4#)

최종적으로 Softmax의 활성화 함수를 통과하게 되면, 총합이 1이 되는 일정한 값으로 변환하게 된다. 그리고 변환된 값 중에서 가장 큰값을 분류의 대상으로 선정하게된다. 위에서 볼 수 있는 것처럼 Dog 0.95 Cat 0.05로 총합이 1인 값으로 변환되었다. 그리고 0.95로 Dog 값이 추출됐기 때문에, 입력 이미지의 값은 Dog로 분류하게 된다. 참고로 케라스는 데이터가 0~1까지의 값으로 변환했을때 최적의 성능을 보인다. 

<br/>
<br/>

# 컨볼루션 신경망 (CNN) 활용 코드
----
```python
#모델 프레임 설정
model = Sequential()
model.add(Dense(50, input_dim=200, activation='relu')) #입력값 200개, 은닉층 50개
model.add(Dense(5, activation='softmax')) # 클래스 5개로 출력

#모델 실행 환경 설정
model.compile(loss='categorcial_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

#모델 실행
history = model.fit(X_train, Y_trian, valiation_data=(X_test, Y_test), epoch=20, batch_size=100, verbose=0) #샘플 
100개로 모두 20번 실행
```
