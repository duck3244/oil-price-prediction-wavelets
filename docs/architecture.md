# 시스템 아키텍처 (Architecture)

Oil Price Prediction with Wavelets 프로젝트의 시스템 아키텍처 문서입니다.

## 1. 개요

본 프로젝트는 **이산 웨이블릿 변환(DWT)** 을 통한 시계열 분해와 **딥러닝(LSTM 계열)** 모델을 결합하여 원유 선물가(WTI, Brent)를 예측하고, 리스크 분석 및 트레이딩 시그널을 제공하는 **3-tier 시스템**입니다.

- **Backend**: Python 기반 ML 코어 + FastAPI REST 서버
- **Frontend**: Vue 3 SPA (Vite + TypeScript + Pinia)
- **Integration**: HTTP/JSON (REST)

## 2. 전체 구조 (Layered View)

```
┌───────────────────────────────────────────────────────────────────┐
│                    Presentation Layer (Frontend)                   │
│  Vue 3 SPA · Vite · Pinia · Vue Router · Tailwind · Chart.js       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐    │
│  │ AnalyzeView │  │ DashboardView│  │  PriceChart (Chart.js) │    │
│  └──────┬──────┘  └──────┬───────┘  └────────────────────────┘    │
│         └─── Pinia Store (usePredictionStore) ───┐                 │
│                                                  │                 │
│                                          axios (client.ts)         │
└──────────────────────────────────────────────────┼─────────────────┘
                                                   │ HTTP / JSON
                                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                             │
│  backend/api/                                                      │
│  ┌────────────────────┐  ┌───────────────────────────────────┐    │
│  │ routers/health.py  │  │ routers/predictions.py            │    │
│  │   GET /api/health  │  │   GET  /api/wavelets              │    │
│  │                    │  │   POST /api/predict               │    │
│  └────────────────────┘  └────────────────┬──────────────────┘    │
│           │                                │                       │
│           ▼                                ▼                       │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  services/prediction_service.py                            │    │
│  │    run_prediction() · list_wavelets()                      │    │
│  └────────────────────────────┬──────────────────────────────┘    │
│  schemas.py: PredictRequest · PredictResponse · HealthResponse     │
└───────────────────────────────┼────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    Domain / ML Core (Python)                       │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │             PredictionEngine (orchestrator)               │     │
│  │   wavelet / level / seq_len / horizon 조율                │     │
│  └────┬──────────────┬──────────────┬──────────────┬────────┘     │
│       │              │              │              │               │
│       ▼              ▼              ▼              ▼               │
│  ┌─────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐     │
│  │  Data   │  │   Wavelet    │  │  Model   │  │    Risk     │     │
│  │Processor│  │  Analyzer    │  │ Builder  │  │  Analyzer   │     │
│  └────┬────┘  └──────┬───────┘  └────┬─────┘  └──────┬──────┘     │
│       │              │               │               │             │
│       ▼              ▼               ▼               ▼             │
│   yfinance        PyWavelets    TensorFlow 2.x     scipy          │
│   sklearn                       (LSTM / Attn /                     │
│                                  CNN-LSTM / Ens.)                  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  TradingSignalGenerator · PredictionVisualizer            │     │
│  └──────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                     Data / Model Persistence                       │
│   yfinance API · data/data.csv · *.h5 체크포인트                   │
└───────────────────────────────────────────────────────────────────┘
```

## 3. 컴포넌트 상세

### 3.1 Frontend (`frontend/`)

| 파일 | 역할 |
|------|------|
| `src/main.ts` | Vue 앱 부트스트랩, Pinia·Router 등록 |
| `src/App.vue` | 루트 레이아웃, 네비게이션 + `<RouterView />` |
| `src/router/index.ts` | 라우트: `/` → Dashboard, `/analyze` → Analyze |
| `src/api/client.ts` | axios 인스턴스(baseURL `/api`, timeout 600s) + 타입 정의 + `predict()`, `getHealth()` |
| `src/stores/prediction.ts` | Pinia 스토어: `result`, `loading`, `error`, `runPrediction()` |
| `src/views/AnalyzeView.vue` | 예측 파라미터 입력 폼 → `store.runPrediction()` |
| `src/views/DashboardView.vue` | 예측 결과 요약 + 차트 표시 |
| `src/components/PriceChart.vue` | Chart.js 기반 과거가·예측가 라인차트 |

**통신 규약**: `VITE_API_BASE_URL` 환경변수 또는 기본 `/api` prefix. 학습 시간이 길어 요청 타임아웃은 10분.

### 3.2 API Layer (`backend/api/`)

- **Framework**: FastAPI + Uvicorn, Pydantic v2 스키마
- **CORS**: `http://localhost:5173` (Vite 개발 서버) 허용
- **라우터 구조**:
  - `routers/health.py` → `GET /api/health`: TF 버전, GPU 사용 가능 여부
  - `routers/predictions.py` → `GET /api/wavelets`, `POST /api/predict`
- **서비스 계층**: `services/prediction_service.py`가 `PredictionEngine`을 호출하고 `numpy → list` 변환 후 `PredictResponse` 반환. 기본적으로 최근 **252 거래일**만 히스토리로 반환.

### 3.3 Domain / ML Core (`backend/*.py`)

| 모듈 | 핵심 클래스 | 책임 |
|------|-------------|------|
| `data_processor.py` | `DataProcessor` | yfinance 수집, 합성 데이터 fallback, 시퀀스 생성, MinMax 스케일링, 기초 통계·이상치 탐지 |
| `wavelet_analyzer.py` | `WaveletAnalyzer` | `pywt.wavedec` 기반 다단 DWT, 컴포넌트 복원·검증, 디노이징, 웨이블릿 비교 |
| `model_builder.py` | `ModelBuilder` | Keras 모델 팩토리 (simple LSTM / BiLSTM / CNN-LSTM / Attention-LSTM / Ensemble / AdvancedEnsemble), 학습 유틸 |
| `predictor_engine.py` | `PredictionEngine` | 전체 파이프라인 오케스트레이터: data → wavelet → per-component train → multi-step predict → reconstruct |
| `risk_analyzer.py` | `RiskAnalyzer`, `TradingSignalGenerator` | 변동성·VaR/CVaR·drawdown·예측 리스크, BUY/SELL/HOLD 시그널·포지션 사이징·SL/TP |
| `visualization_tools.py` | `PredictionVisualizer` | Matplotlib/Seaborn 기반 가격·분해·학습·리스크·시그널 시각화 |
| `main.py` | `OilPricePredictionApp` | CLI 엔트리포인트 (basic / advanced / comparison 모드) |

### 3.4 설정 파일 (`backend/config_example.json`)

컴포넌트별 모델 선택은 config로 가변:

```json
{
  "wavelet": "db4",
  "decomposition_level": 5,
  "sequence_length": 60,
  "prediction_horizon": 1,
  "model_config": {
    "trend":    "ensemble",
    "detail_1": "bidirectional",
    "detail_2": "cnn_lstm",
    "detail_3": "attention",
    "detail_4": "simple",
    "detail_5": "simple"
  }
}
```

## 4. 데이터 플로우 (End-to-End)

```
  [yfinance API / CSV]
          │
          ▼
 ┌──────────────────────┐
 │  DataProcessor       │  fetch_oil_data() → OHLCV DataFrame
 │                      │  (실패 시 synthetic data fallback)
 └──────────┬───────────┘
            ▼  oil_prices: 1D ndarray
 ┌──────────────────────┐
 │  WaveletAnalyzer     │  decompose(level=5)
 │                      │  → {trend, detail_1 … detail_5}
 └──────────┬───────────┘
            ▼ 컴포넌트별 신호
 ┌──────────────────────┐
 │  DataProcessor       │  create_sequences() per component
 │  (per-component      │  → X:(N, seq_len, 1), y:(N, horizon)
 │   MinMaxScaler)      │  sequential split 80/10/10
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │  ModelBuilder        │  config에 따라 컴포넌트별 모델 생성
 │                      │  Adam(lr=1e-3), Huber loss
 │                      │  EarlyStopping(20) + ReduceLROnPlateau(10)
 └──────────┬───────────┘
            ▼ trained models + histories
 ┌──────────────────────┐
 │  PredictionEngine    │  predict_next_values(n_steps) per component
 │  (iterative forecast)│  inverse_transform → component predictions
 │                      │  reconstruct_predictions = Σ components
 └──────────┬───────────┘
            ▼ final forecast
 ┌──────────────────────┐       ┌────────────────────────────┐
 │  RiskAnalyzer        │──────▶│ TradingSignalGenerator     │
 │  volatility·VaR·CVaR │       │ BUY/SELL/HOLD · sizing     │
 │  drawdown · CI       │       │ stop-loss / take-profit    │
 └──────────┬───────────┘       └────────────┬───────────────┘
            ▼                                 ▼
                  PredictResponse (JSON) → Frontend
```

## 5. 모듈 의존성

```
main.py ──┬─▶ data_processor
          ├─▶ wavelet_analyzer
          ├─▶ model_builder
          ├─▶ predictor_engine ──┬─▶ data_processor
          │                      ├─▶ wavelet_analyzer
          │                      └─▶ model_builder
          ├─▶ risk_analyzer
          └─▶ visualization_tools

api/app.py ──┬─▶ routers/health ─────────────▶ schemas
             └─▶ routers/predictions ─────────▶ schemas
                                      └─────▶ services/prediction_service
                                                    ├─▶ predictor_engine
                                                    └─▶ wavelet_analyzer
```

## 6. 주요 기술 스택

| 계층 | 기술 |
|------|------|
| ML Core | TensorFlow ≥ 2.10, NumPy, Pandas, scikit-learn, PyWavelets, SciPy |
| Data | yfinance (Yahoo Finance) |
| Viz | Matplotlib, Seaborn |
| API | FastAPI ≥ 0.110, Uvicorn, Pydantic v2 |
| Frontend | Vue 3, Vite 6, TypeScript 5, Pinia, Vue Router 4, axios, Chart.js 4 + vue-chartjs, Tailwind 3 |

## 7. 실행 방법

**Backend API**
```bash
cd backend
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

**CLI (배치 예측)**
```bash
cd backend
python main.py --mode basic --symbol CL=F --days 30
```

**Frontend**
```bash
cd frontend
npm install
npm run dev     # http://localhost:5173
```

## 8. 설계상 주요 결정

- **컴포넌트별 독립 스케일러**: 각 웨이블릿 컴포넌트(trend / details)는 진폭 레벨이 달라 하나의 스케일러를 공유하면 detail이 묻힘 → 컴포넌트마다 별도 `MinMaxScaler` 사용.
- **컴포넌트별 이질 모델**: 저주파(trend)는 복잡한 앙상블, 고주파 detail은 얕은 네트워크로 — 각 주파수 대역의 신호 특성에 모델 용량을 매칭.
- **Multi-step 반복 예측**: horizon=1로 학습한 후, 예측값을 시퀀스에 밀어넣으며 n_steps 반복. 단일 스텝 학습의 안정성과 다중 스텝 forecasting을 양립.
- **10분 요청 타임아웃**: 학습이 포함된 동기 REST 호출. 프로덕션화 시에는 비동기 잡 큐(Celery·RQ)로 분리 권장.
- **CORS 화이트리스트**: Vite dev 서버 `:5173`만 허용하는 최소 권한 원칙.
