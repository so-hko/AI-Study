# Train Set과 Validation Set, 그리고 Test Set

우리가 딥러닝 모델을 만들고 학습하고 성능평가를 하고 배포까지 한다고 할 때,
데이터를 Train data set, Validation data set, Test data set으로 나누는 과정이 중요하다.
<br>
나누기 위해 비율설정부터 학습, 검증, 테스트의 역할까지 헷갈릴 수 있다는 생각이 들어서 정리하고자 한다.
<br><br>


----------------------------------------------------------------------

### ◆ 데이터 나누는 비율(split ratio)

내마음대로 나누면된다...보통 train : valid : test 기준으로 6:2:2 또는 5:3:2 라고 하는데...<br>
데이터 수가 많지 않으면 그에 맞게 학습비율을 더 높여야하지않을까? 즉 자기 맘..ㅎㅎ

### ◆ Train set
Train은 해석 그대로 학습이다. 모델 "학습"을 하기 위한 데이터

###  ◆ Validation과 Test Data set
많이 Valid와 Test set을 헷갈려한다. 나도 참 헷갈려했다..<br>
Valid는 검증이라는 뜻이고 Test는 말그대로 테스트!<br>
보통 Valid는 모델 학습이후에 이 모델이 잘 학습이 되었는지 "검증"을 하기 위해 많이 사용되고 모델 내부에 Validation_split 비율을 써서 train data set내에서 자르는 경우도 있다.<br>
그래서 헷갈리는 경우가 많은 것도 같다. 보통 만들어진 모델을 쓸때 train, valid, test로 나누기보다는 train, test로 (주로 80:20 비율로)나누곤 하니까.<br>
때때로 train, test로 먼저나누고 모델 내부에서 train데이터에서 validation set을 구축해서 쓰는경우도 많다 (예:cross validation)
<br><br>
그리고 검증(valid)결과로 하이퍼파라미터를 계속 튜닝해주면서 retrain을 하면서 모델 성능을 끌어올리고 과적합이 되지는 않았는지...뭐...진짜 이 모델 성능이 정말 잘 나오는지를 최종적으로 하기 위해 test dataset을 사용한다.




