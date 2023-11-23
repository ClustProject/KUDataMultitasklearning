# Time Series Multi-task learning

## 1. BeijingPM Dataset 에서 해당 데이터셋 을 받아, Trainer.ipynb 을 통해 pickle file 로 저장함

- 입력 데이터 형태 : (num_of_instance, input_dims, time_steps) 차원의 단변량시계열 데이터
  <br>

**Time series Representation 사용 시, 설정해야하는 값**

- **best_model_path** : 학습 완료된 모델을 저장할 경로

- **시계열 분류 모델 hyperparameter :** 아래에 자세히 설명.
  - LSTM_FCNs hyperparameter
    <br>

#### 시계열 분류 모델 hyperparameter <br>
