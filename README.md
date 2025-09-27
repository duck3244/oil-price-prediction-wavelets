# 🛢️ Oil Price Prediction using Wavelets and TensorFlow 2.0

현대적인 웨이블릿 분해와 딥러닝 기법을 결합한 유가 예측 시스템입니다.

## 🌟 주요 특징

### 🔬 기술적 특징
- **웨이블릿 분해**: Discrete Wavelet Transform으로 가격 신호 분해
- **딥러닝 모델**: TensorFlow 2.0 기반 다양한 LSTM 아키텍처
- **앙상블 방법**: 여러 모델 조합으로 예측 정확도 향상
- **실시간 데이터**: Yahoo Finance API 연동
- **종합적 평가**: MSE, MAE, RMSE, R², MAPE 등 다양한 메트릭

### 📊 분석 기능
- **위험 분석**: VaR, CVaR, 변동성 분석
- **트레이딩 신호**: 매수/매도 추천 및 포지션 사이징
- **시각화**: 종합적인 차트와 대시보드
- **모델 해석**: 구성요소별 기여도 분석

## 🚀 빠른 시작

### 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 또는 개발 모드로 설치
pip install -e .
```

### 기본 사용법

```python
# 기본 예측 실행
python main.py --mode basic

# 고급 분석 실행
python main.py --mode advanced

# Brent 유가 예측
python main.py --symbol BZ=F --days 60

# 웨이블릿 비교 분석
python main.py --mode comparison
```

### 프로그래밍 방식 사용

```python
from predictor_engine import PredictionEngine
from risk_analyzer import RiskAnalyzer, TradingSignalGenerator

# 예측 엔진 초기화
engine = PredictionEngine(
    wavelet='db4',
    decomposition_level=5,
    sequence_length=60
)

# 전체 파이프라인 실행
results = engine.run_full_pipeline(
    symbol="CL=F",
    n_predictions=30
)

# 위험 분석
risk_analyzer = RiskAnalyzer()
risk_report = risk_analyzer.generate_risk_report(
    historical_prices, predictions
)

# 트레이딩 신호 생성
signal_gen = TradingSignalGenerator(risk_tolerance='medium')
signals = signal_gen.generate_comprehensive_signals(
    current_price, predictions, risk_report
)
```

## 📁 프로젝트 구조

```
oil-price-prediction-wavelets/
├── main.py                    # 메인 실행 파일
├── data_processor.py          # 데이터 처리 모듈
├── wavelet_analyzer.py        # 웨이블릿 분석 모듈  
├── model_builder.py           # 모델 구축 모듈
├── predictor_engine.py        # 예측 엔진 모듈
├── risk_analyzer.py           # 리스크 분석 모듈
├── visualization_tools.py     # 시각화 도구 모듈
├── requirements.txt           # 의존성 목록
├── setup.py                  # 패키지 설정
└── README.md                 # 프로젝트 문서
```

## 🔧 모듈 설명

### 1. `data_processor.py`
- 유가 데이터 수집 (Yahoo Finance)
- 데이터 전처리 및 시퀀스 생성
- 기술적 지표 계산
- 통계적 분석

### 2. `wavelet_analyzer.py` 
- 웨이블릿 분해 (다양한 웨이블릿 지원)
- 구성요소 분석 및 재구성
- 신호 잡음 제거
- 웨이블릿 비교 분석

### 3. `model_builder.py`
- LSTM 모델 아키텍처 (Simple, Bidirectional, CNN-LSTM, Attention)
- 앙상블 모델 구축
- 모델 훈련 및 평가
- 하이퍼파라미터 최적화

### 4. `predictor_engine.py`
- 전체 예측 파이프라인 조정
- 구성요소별 모델 훈련
- 예측 생성 및 재구성
- 모델 저장/로드 기능

### 5. `risk_analyzer.py`
- 변동성 분석 (일간, 연간, 롤링)
- VaR/CVaR 계산
- 드로다운 분석
- 트레이딩 신호 생성
- 포지션 사이징

### 6. `visualization_tools.py`
- 가격 예측 차트
- 웨이블릿 분해 시각화
- 위험 분석 대시보드
- 트레이딩 신호 대시보드
- 모델 성능 비교

## 📊 사용 예제

### 예제 1: 기본 예측

```python
from main import OilPricePredictionApp

app = OilPricePredictionApp()

# WTI 유가 30일 예측
results = app.run_basic_prediction(
    symbol="CL=F",
    days_ahead=30,
    plot_results=True
)

print(f"현재 가격: ${results['predictions']['current_price']:.2f}")
print(f"내일 예측: ${results['predictions']['predictions'][0]:.2f}")
```

### 예제 2: 고급 분석

```python
# 사용자 정의 설정
config = {
    'wavelet': 'db8',
    'decomposition_level': 6,
    'sequence_length': 120,
    'prediction_days': 60,
    'model_config': {
        'trend': 'advanced_ensemble',
        'detail_1': 'attention',
        'detail_2': 'bidirectional'
    },
    'risk_analysis': {
        'risk_tolerance': 'high'
    }
}

# 고급 분석 실행
results = app.run_advanced_analysis(
    symbol="BZ=F",  # Brent 유가
    config=config
)
```

### 예제 3: 웨이블릿 비교

```python
# 여러 웨이블릿 성능 비교
comparison = app.run_comparison_analysis(
    symbols=["CL=F", "BZ=F"],
    wavelets=['db4', 'db8', 'haar', 'bior2.2', 'coif3']
)

# 최적 웨이블릿 확인
for symbol, results in comparison.items():
    best_wavelet = min(results.keys(), 
                      key=lambda k: results[k].get('avg_validation_loss', float('inf')))
    print(f"{symbol} 최적 웨이블릿: {best_wavelet}")
```

## 🎯 모델 아키텍처

### 웨이블릿 분해
```
원본 신호 → [웨이블릿 분해] → 트렌드 + 세부성분들
                                ↓
                         개별 LSTM 모델들
                                ↓
                         구성요소 예측들 → [재구성] → 최종 예측
```

### LSTM 모델 종류

1. **Simple LSTM**: 기본적인 LSTM 구조
2. **Bidirectional LSTM**: 양방향 정보 처리
3. **CNN-LSTM**: 컨볼루션과 LSTM 조합
4. **Attention LSTM**: 어텐션 메커니즘 적용
5. **Ensemble**: 여러 모델의 앙상블
6. **Advanced Ensemble**: 학습된 가중치 앙상블

## 📈 성능 메트릭

### 예측 정확도
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error
- **R²**: 결정계수
- **MAPE**: Mean Absolute Percentage Error

### 위험 메트릭
- **VaR**: Value at Risk (95% 신뢰구간)
- **CVaR**: Conditional Value at Risk
- **변동성**: 일간/연간 변동성
- **최대 드로다운**: 최대 손실 구간
- **샤프 비율**: 위험 조정 수익률

## 🛡️ 위험 분석

### 변동성 분석
- 30/60/90일 롤링 변동성
- 지수가중 이동평균 변동성
- 변동성 클러스터링 분석
- 변동성 백분위수 분석

### VaR 계산 방법
- **Historical VaR**: 과거 데이터 기반
- **Parametric VaR**: 정규분포 가정
- **Modified VaR**: Cornish-Fisher 보정
- **Conditional VaR**: 기대손실

## 📊 트레이딩 신호

### 신호 생성
- **방향성 신호**: BUY/SELL/HOLD
- **강도**: STRONG/MODERATE/WEAK
- **신뢰도**: HIGH/MEDIUM/LOW
- **시간대별**: 1일/1주/1개월 전망

### 포지션 사이징
- **위험 기반**: 포트폴리오 대비 위험 비율
- **변동성 조정**: 현재 변동성 반영
- **Kelly 기준**: 수학적 최적 비율
- **VaR 기반**: 손실 위험 제한

### 손절/익절 설정
- **동적 레벨**: 변동성 기반 자동 조정
- **위험-수익 비율**: 최소 1:2 권장
- **트레일링 스탑**: 수익 보호 메커니즘

## 🎨 시각화 기능

### 차트 종류
- **가격 예측 차트**: 과거 + 미래 가격
- **웨이블릿 분해**: 구성요소별 분해 결과
- **위험 대시보드**: 종합 위험 분석
- **트레이딩 시그널**: 매매 신호 시각화
- **모델 성능**: 훈련 결과 비교

### 대시보드 구성
- **실시간 업데이트**: 최신 데이터 반영
- **인터랙티브**: 확대/축소, 필터링
- **다중 시간대**: 일간/주간/월간 뷰
- **알림 시스템**: 중요 신호 하이라이트

## ⚙️ 설정 및 커스터마이징

### 웨이블릿 설정
```python
# 사용 가능한 웨이블릿들
wavelets = {
    'Daubechies': ['db1', 'db2', ..., 'db20'],
    'Biorthogonal': ['bior1.1', 'bior2.2', ...],
    'Coiflets': ['coif1', 'coif2', ...],
    'Haar': ['haar'],
    'Symlets': ['sym2', 'sym3', ...]
}
```

### 모델 설정
```python
model_config = {
    'trend': 'ensemble',        # 트렌드용 모델
    'detail_1': 'bidirectional', # 고주파 성분용
    'detail_2': 'cnn_lstm',     # 중주파 성분용
    'detail_3': 'attention',    # 저주파 성분용
    'detail_4': 'simple',       # 기타 성분용
    'detail_5': 'simple'
}
```

### 훈련 설정
```python
training_config = {
    'epochs': 100,
    'batch_size': 32,
    'train_ratio': 0.8,
    'validation_ratio': 0.1,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10
}
```

## 🔄 데이터 소스

### 지원하는 심볼
- **CL=F**: WTI 원유 선물
- **BZ=F**: Brent 원유 선물
- **HO=F**: 난방유 선물
- **RB=F**: 휘발유 선물

### 데이터 주기
- **일간**: 기본 분석 단위
- **실시간**: Yahoo Finance API
- **과거 데이터**: 2010년부터 현재까지

## 🛠️ 문제 해결

### 자주 발생하는 문제들

#### 데이터 수집 오류
```bash
# 인터넷 연결 확인
ping finance.yahoo.com

# 대체 심볼 시도
python main.py --symbol BZ=F
```

#### 메모리 부족
```python
# 배치 크기 줄이기
config['training']['batch_size'] = 16

# 시퀀스 길이 줄이기  
config['sequence_length'] = 30
```

#### GPU 메모리 부족
```python
# TensorFlow GPU 메모리 증가 허용
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### 성능 최적화

#### CPU 최적화
```python
# 멀티스레딩 설정
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
```

#### 훈련 속도 향상
```python
# Mixed Precision 사용
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## 📚 참고 자료

### 기술 문서
- [TensorFlow 2.0 Documentation](https://www.tensorflow.org/)
- [PyWavelets Documentation](https://pywavelets.readthedocs.io/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
