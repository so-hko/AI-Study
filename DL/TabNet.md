# [Paper Review] TabNet : Attentive Interpretable Tabular Learning (2021)

## 1. Introduction

    
Deep Neural Network(DNN) 은 Image나 Audio와 같은 비정형 데이터에서 좋은 성능을 보이며 많은 주목을 받고 있다.
그러나, 표(TABLE)의 형태를 띄는 정형 데이터인 Tabular Data에 한정해서는 DNN보다 Ensemble Tree기반 머신러닝 방법론들이 더 좋은 성능을 보이고 많이 사용된다.

하지만, DNN기반의 Tabular Dataset을 위한 방법은 필요하다. 
그 이유는 대규모의 데이터세트가 들어왔을 때 DNN기반 방법들이 확률통계적 방법보다 더 좋은 성능을 보일 수 있기 때문이다. 

그리고 본 논문에서 제안하는 TabNet은 4가지의 특징을 갖는다.
    
```
    1. 전처리 과정이 따로 필요하지 않고 Grandient Descent 기반 Optimization을 통해 end-to-end learning이 가능
    2. 각각의 Decision step에서 Sequential attention을 사용하여 Feature selection 을 진행
    3. 위 단계를 통해 두가지의 특성을 얻을 수 있음
        (1) 다른 학습 모델보다 더 좋은 분류/회귀 예측 결과를 보임
        (2) Local interpretability : 
            feature importance와 feature가 어떻게 combine되었는지를 visualize(시각화)함
            Global interpretability : 
            각 feature들의 영향/기여도를 수치화 함
    4. masked feature값을 예측하기 위해 unsupervised pre-training단계를 진행하여 높은 성능향상을 보임
```

## 2. Related Work

#### 1. Feature Selection
#### 2. Tree-based Learning
#### 3. Self-supervised Learning


## 3. TabNet for Tabular Learning

![TabNet_overall_Architecture](https://github.com/so-hko/Study/blob/main/DL/images/TabNet%20overall%20architecture.png?raw=true)
Attentive Interpretable Tabular Learning인 TabNetwork의 전반적인 구조는 위와 같다.<br>
위 아키텍처는 Step1부터 StepN까지 구성되고, 각각의 Step마다 (1) Feature transformer와 (2) Attentive transformer, (3) Mask 처리를 해준다. 
특히 그림a에서 볼 수 있듯이 Feature transformer block 부분을 처리한 후 나오는 결과(차후에 더 자세히 설명예정)가 split block으로 가면 
split block은 차후에 진행될 단계인 attentive transformer 단계로도 넘겨주고 Relu활성화 함수와 함께 다음 스텝으로 넘겨주어 
최후의 overall output 도출<span style="color:gray">(나는 이걸 예측(prediction)결과로 이해했다.)</span>을 위해 
representation을 2개로 나누어(split) 주는 역할을 한다. 
Mask Block은 Attentive transformer block에서 정보를 받아 각각의 step에서 feature selection을 위해 사용된 마스킹(Masking)정보를 담고 있다. 
그리고 이 모든 스텝마다의 마스킹 정보들은 추후에 Agg. Block을 통해 aggregate(집계)되어 최종적으로 어떤 feature attributes들이 사용되었는지 정보를 알 수 있다.<br>
모델에 대한 더 정확하고 자세한 이해를 위해 모델 아키텍처의 구성요소들과 Input, Output 및 동작원리 등에 대해 살펴보겠다.<br><br>
① INPUT <br>
　 모델의 입력으로는 Tabular Data가 들어간다. Tabular Data는 표(table)의 형태를 띄우는 데이터를 의미하는데 흔히 엑셀에서 표현될 수 있는 데이터라고 생각하면 된다. 
이 Tabular 데이터는 예측모델설계를 위해 Numerical과 Categorical 종류로 나뉠 수 있는데, Numerical의 경우에는 label값이 수치적으로 표현되는, 예를들어 집값데이터, 주식데이터 처럼 0.1,0.3...과 같은 수치로 표현되는 데이터를 의미하고,
Categorical 데이터의 경우에는 동물(개,고양이)이라던가 나라(프랑스, 한국, 미국, 일본), 혹은 상태(찌그러짐, 평평한, 갈라짐)과 같은 카테고리로 나누어지는 feature들을 다루는 경우를 의미한다. 
caterical feature의 경우 원핫인코딩과 같은 인코딩방법을 통해 수치화해주는 처리가 필요로 되는데, 
본 논문에서 제안하는 TabNet 모델에서는 Categorical variable 또한 별다른 전처리를 하지않아도 되도록 임베딩해주는 레이어를 구성한다.
또한 Input으로 Tabular Data가 들어오면 바로 BN(BatchNormalization)레이어를 거쳐 ~을 위해 Feature transformer Block으로 들어가서 처리된다.

② Feature Transformer <br>



## 4. Experiments

## 5. Conclusions
