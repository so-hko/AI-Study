# [Paper Review] TabNet : Attentive Interpretable Tabular Learning (2021)

## 1. Introduction

    
Deep Neural Network(DNN) 은 Image나 Audio와 같은 비정형 데이터에서 좋은 성능을 보이며 많은 주목을 받고 있다.
그러나, 표(TABLE)의 형태를 띄는 정형 데이터인 Tabular Data에 한정해서는 DNN보다 Ensemble Tree기반 머신러닝 방법론들이 더 좋은 성능을 보이고 많이 사용된다.

하지만, DNN기반의 Tabular Dataset을 위한 방법은 필요하다. 그 이유는 대규모의 데이터세트가 들어왔을 때 DNN기반 방법들이 확률통계적 방법보다 더 좋은 성능을 보일 수 있기 때문이다. 

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

![TabNet overall Architecture] (https://github.com/so-hko/Study/blob/main/DL/images/TabNet%20overall%20architecture.png?raw=true)


## 4. Experiments

## 5. Conclusions
