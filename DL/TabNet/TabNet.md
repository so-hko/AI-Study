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
<br>
Attentive Interpretable Tabular Learning인 TabNetwork의 전반적인 구조는 위와 같다.<br>
위 아키텍처는 Step1부터 StepN까지 구성되고, 각각의 Step마다 (1) Feature transformer와 (2) Attentive transformer, (3) Mask 처리를 해준다. 
특히 그림a에서 볼 수 있듯이 Feature transformer block 부분을 처리한 후 나오는 결과(차후에 더 자세히 설명예정)가 split block으로 가면 
split block은 차후에 진행될 단계인 attentive transformer 단계로도 넘겨주고 Relu활성화 함수와 함께 다음 스텝으로 넘겨주어 
최후의 overall output 도출 <span style="color:#D3D3D3">(나는 이걸 예측(prediction)결과로 이해했다.) </span> 을 위해 
representation을 2개로 나누어(split) 주는 역할을 한다. 
Mask Block은 Attentive transformer block에서 정보를 받아 각각의 step에서 feature selection을 위해 사용된 마스킹(Masking)정보를 담고 있다. 
그리고 이 모든 스텝마다의 마스킹 정보들은 추후에 Agg. Block을 통해 aggregate(집계)되어 n번째 step이후 최종적으로 어떤 feature attributes들이 사용되었는지 정보를 알 수 있다.<br>
모델에 대한 더 정확하고 자세한 이해를 위해 모델 아키텍처의 구성요소들과 Input, Output 및 동작원리 등에 대해 살펴보겠다.<br><br>
① INPUT <br>
　 모델의 입력으로는 Tabular Data가 들어간다. Tabular Data는 표(table)의 형태를 띄 우는 데이터를 의미하는데 흔히 엑셀에서 표현될 수 있는 데이터라고 생각하면 된다. 
이 Tabular 데이터는 예측모델설계를 위해 Numerical과 Categorical 종류로 나뉠 수 있는데, Numerical의 경우에는 label값이 수치적으로 표현되는, 예를들어 집값데이터, 주식데이터 처럼 0.1,0.3...과 같은 수치로 표현되는 데이터를 의미하고,
Categorical 데이터의 경우에는 동물(개,고양이)이라던가 나라(프랑스, 한국, 미국, 일본), 혹은 상태(찌그러짐, 평평한, 갈라짐)과 같은 카테고리로 나누어지는 feature들을 다루는 경우를 의미한다. 
caterical feature의 경우 원핫인코딩과 같은 인코딩방법을 통해 수치화해주는 처리가 필요로 되는데, 따라서
본 논문에서 실험할 때 Categorical variable를 가지는 데이터를 실험하는 경우 임베딩해주는 레이어를 구성하였다.
또한 Input으로 Tabular Data가 들어오면 바로 BN(BatchNormalization)레이어를 거쳐 Feature transformer Block으로 들어가서 처리된다. 

② Feature Transformer <br><br>
![Feature Transformer](https://github.com/so-hko/Study/blob/main/DL/images/FeatureTransformer.png?raw=true) <br>
위 그림에서 우리는 Feature Transformer Block 처리과정을 보고 이해할 수 있다. 
<span style="color:#D3D3D3"> 2022/1/26솔직히 Feature Transformer 블록 처리를 왜 해주는 건지에 대해서는 아직 정확히 이해가 되지않아 글로 표현이 안된다.
(논문을 좀더 살펴봐야겠다...ㅠㅠ)</span> <br>
(2022/1/27) 2개의 shared layer와 2개의 decision step-dependent layer을 잇기(concatenation)위한 특징들을 얻기 위해 feature transformer block을 사용하여 선택된 변수들을 처리한다.
그림에서 알 수 있듯이 FC(Fully Connected Layer) - BN(Batch Normalization) - GLU(Gated Linear Unit)를 4번 반복하는 구조를 가지고 있다. <br>
(1) Shared across decision steps : 그 중 2개 layer들은 모든 전체 overall architecture에서 진행되는 모든 Step들과 공유되고, <br> 
(2) Decision step dependent : 나머지 2개 layer집합은 현재 진행되고 있는 해당 Step에서만 사용되는 layer이다. <br>

③ Attentive transformer <br><br>
<p align="center"><img src="https://github.com/so-hko/Study/blob/main/DL/images/AttentiveTransformer.png?raw=true" height="300px" width="300"> <br> </p>
<br>
 Attentive transformer Block은 위 그림과 같이 구성된다. 이전 Decision step에서 각 feature가 얼마나 많이 사용되었는지를 집계한 정보를 Prior scales 에 저장한다.
또한 이전 단계들의 중요한 Feature를 Selection하기 위해 sparsemax 활성화 함수를 사용한다.<br>
Sparsemax는 softmax보다 sparsity(희소성)을 강조하기 위해 제안된 activate function인데 이를 통해 극단적인 드라마틱한 Feature Selection 효과를 얻을 수 있고,
sparse matrix와 변수들을 element-wise하면 특정 변수만 선택한 효과를 얻을 수 있다.

<br> 

④ Output <br>
아웃풋은 Step 1부터 Step N까지를 모두 거쳐 나온 예측(Prediction)과 각각의 Step마다 존재하는 Attentive tranformer와 Feature transformer 단계들을 거처 Feature Selection과 Feature Masking을 해준 결과를 바탕으로 가장 많이 쓰인(유의미한) Feature attributes를 아웃풋으로 생각할 수 있을 것 같다.

## 4. Experiments

## 5. Conclusions

---------------------------------------------------
2022년 01월 26일 : 논문 나름 꼼꼼히 읽어봤다 생각했는데 막상 정리하려고 하니 하나도 개념이 구조적으로 정립되지 않았다는 생각이 들었다.
아직 feature transformer의 역활 어떤 형태로 output이 나오는지 등에 대해 잘 이해가 되지 않는 듯 하다.(1월27일 :  다시읽어보니까 다음 step을 위해 결과물과 정보들의 결정 단계를 분리하는 특징들을 얻기 위해 feature transformer을 사용한 듯) 논문을 더 살펴보고 제공되어있는 코드로 예제/실험 을 한번 해봐야겠다.

TabNet논문저자 깃허브에 [코드](https://github.com/dreamquark-ai/tabnet) 가 올려져있으니 이것을 참고해보는 것도 좋을듯 싶다.

```python

```
