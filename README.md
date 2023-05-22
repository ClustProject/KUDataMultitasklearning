# Time Series Transfer learning

## 1. UCR Dataset 에서 해당 데이터셋 을 받아, Set_Pickle.ipynb 을 통해 pickle file 로 저장함

- 입력 데이터 형태 : (num_of_instance, input_dims, time_steps) 차원의 단변량시계열 데이터
<br>

**Time series Representation 사용 시, 설정해야하는 값**
* **best_model_path** : 학습 완료된 모델을 저장할 경로


* **시계열 분류 모델 hyperparameter :** 아래에 자세히 설명.
  * LSTM_FCNs hyperparameter
<br>

#### 시계열 분류 모델 hyperparameter <br>

#### LSTM-FCNs (w/o data representation)
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **num_layers** : recurrent layers의 수, int(default: 1, 범위: 1 이상)
- **lstm_drop_out** : LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
- **fc_drop_out** : FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)

- **freeze** : fc layer 를 제외한 layer 의 학습 여부 (default: False)

 

