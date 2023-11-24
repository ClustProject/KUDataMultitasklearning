# Time Series Multi-task learning

## 1. Without data representation

- 원본 시계열 데이터를 입력으로 활용하여 학습 (single 모델로 사전학습 → multi-task learning으로 미세조정)

  (1) regression: single regression 모델로 사전학습 후, multitask-Learning 모델로 미세조정 수행

  (2) classificaiton: single classification 단일 모델로 사전학습 후, (regression+classification) 모델로 미세조정 수행

  (3) multi-task learning: 동일한 backbone 구조에 classification head 및 regression head를 추가하여 2개의 task를 동시에 학습


- 입력 데이터 형태 : (num_of_instance, input_dims, time_steps) 차원의 시계열 데이터
  <br>

**[Hyperparmeter for single regression or classification (LSTM & GRU)]**
- **model** : 모델명 (LSTM or GRU)
- **best_model_path** : 학습 완료된 모델을 저장할 경로
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **num_layers** : recurrent layers의 수, int(default: 2, 범위: 1 이상)
- **hidden_size** : hidden state의 차원, int(default: 64, 범위: 1 이상)
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **bidirectional** : 모델의 양방향성 여부, bool(default: True)
- **num_epochs** : 학습 epoch 횟수, int(default: 1000, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 16, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>

**[Hyperparmeter for single regression or classification (1D CNN)]**
- **model** : 모델명 (1D CNN)
- **best_model_path** : 학습 완료된 모델을 저장할 경로
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **seq_len** : 데이터의 시간 길이, int
- **output_channels** : convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
- **kernel_size** : convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
- **stride** : convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
- **padding** : padding 크기, int(default: 0, 범위: 0 이상)
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **num_epochs** : 학습 epoch 횟수, int(default: 1000, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 16, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>

**[Hyperparmeter for single regression or classification (LSTM_FCNs_reg & LSTM_FCNs_cls)]**
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수 (regression이면 1로 지정), int 
- **num_layers** : recurrent layers의 수, int(default: 2, 범위: 1 이상)
- **lstm_drop_out** : LSTM dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **fc_drop_out** : FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **num_epochs** : 학습 epoch 횟수, int(default: 1000, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 16, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
- **freeze** : fc layer 를 제외한 layer 의 학습 여부 (default: False)
<br>

**[Hyperparmeter for multitask-learning (LSTM_FCNs_multi)]**
- **model** : 모델명 (LSTM_FCNs_multi)
- **best_model_path** : 학습 완료된 모델을 저장할 경로
- **input_size** : 데이터의 변수 개수, int
- **num_classes_1** : Task1의 class 개수, int
- **num_classes_2** : Task2의 class 개수 (regression이면 1개), int
- **num_layers** : recurrent layers의 수, int(default: 1, 범위: 1 이상)
- **lstm_drop_out** : LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
- **fc_drop_out** : FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
- **freeze** : fc layer 를 제외한 layer 의 학습 여부 (default: False)
- **alpha** : Task1의 반영 비율 (default: 0.5 (범위: [0, 1]))

## 2. With data representation

- 원본 시계열 데이터를 representation vector로 변환한 데이터를 입력으로 활용하여 학습

- 동일한 Backbone 구조에 Classification 및 Regression Head를 추가하여 2개의 Task를 동시에 학습

- 입력 데이터 형태 : (num_of_instance, embedding_dim) 차원의 시계열 데이터
  <br>

**[Hyperparmeter for FC]**
- **model** : 모델명 (FC)
- **best_model_path** : 학습 완료된 모델을 저장할 경로
- **시계열 모델 hyperparameter :** 아래에 자세히 설명.
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **bias** : bias 사용 여부, bool(default: True)
- **num_epochs** : 학습 epoch 횟수, int(default: 1000, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 16, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>
