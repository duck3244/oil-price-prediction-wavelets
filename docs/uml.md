# UML Diagrams

Oil Price Prediction 프로젝트의 UML 다이어그램 모음입니다. 모든 다이어그램은 Mermaid로 작성되어 있어 GitHub, VS Code, IntelliJ 등에서 바로 렌더링됩니다.

## 1. 클래스 다이어그램 — ML 코어

```mermaid
classDiagram
    class DataProcessor {
        -int sequence_length
        -int prediction_horizon
        -MinMaxScaler scaler
        -ndarray raw_data
        -ndarray oil_prices
        -DatetimeIndex dates
        +fetch_oil_data(symbol, start, end) DataFrame
        +create_sequences(data, target_col, scaler) tuple
        +split_data(X, y, train_ratio, val_ratio) tuple
        +inverse_transform(scaled_data) ndarray
        +get_data_statistics() Dict
        +detect_anomalies(threshold) Dict
        +prepare_prediction_input(n_steps_back) ndarray
        -_generate_synthetic_data(n_points) DataFrame
    }

    class WaveletAnalyzer {
        -str wavelet
        -int decomposition_level
        -Dict coefficients
        -Dict components
        -int original_length
        +decompose(signal, plot) Dict
        +reconstruct_signal(components) ndarray
        +validate_reconstruction(signal, tol) Dict
        +analyze_components(signal) Dict
        +denoise_signal(signal, type, mode) ndarray
        +compare_wavelets(signal, wavelet_list) Dict
        +extract_features(signal) Dict
        +plot_decomposition(signal, dates) void
        -_reconstruct_components(signal) Dict
    }

    class ModelBuilder {
        -int sequence_length
        -int prediction_horizon
        -int seed
        -Dict models
        -Dict histories
        +create_simple_lstm(shape, units, dropout, name) Model
        +create_bidirectional_lstm(shape, units, dropout, name) Model
        +create_cnn_lstm(shape, filters, units, dropout, name) Model
        +create_attention_lstm(shape, units, dropout, name) Model
        +create_ensemble_model(shape, name) Model
        +create_advanced_ensemble(shape, name) Model
        +train_model(model, X_tr, y_tr, X_val, y_val, epochs, batch, save_best) History
        +create_model_by_type(type, shape) Model
        +get_model_summary(name) void
    }

    class PredictionEngine {
        -DataProcessor data_processor
        -WaveletAnalyzer wavelet_analyzer
        -ModelBuilder model_builder
        -Dict components
        -Dict component_models
        -Dict component_scalers
        -Dict component_histories
        -bool is_trained
        +load_and_prepare_data(symbol, start, end) DataFrame
        +decompose_signal(plot) Dict
        +train_component_models(config, train_ratio, val_ratio, epochs, batch) Dict
        +predict_next_values(n_steps) Dict
        +reconstruct_predictions(component_preds) ndarray
        +predict(n_steps) Dict
        +run_full_pipeline(symbol, start, end, n_predictions) Dict
        +save_models(path) void
        +load_models(path) void
        +get_model_performance() Dict
    }

    class RiskAnalyzer {
        -float confidence_level
        -float alpha
        +calculate_volatility_metrics(prices) Dict
        +calculate_var_cvar(prices, holding_period) Dict
        +analyze_prediction_risk(preds, current) Dict
        +calculate_drawdown_metrics(prices) Dict
        +generate_confidence_intervals(preds, levels) Dict
        +assess_model_stability(component_preds) Dict
        +generate_risk_report(prices, preds, component_preds) Dict
    }

    class TradingSignalGenerator {
        -str risk_tolerance
        -Dict risk_multipliers
        +generate_directional_signals(current, preds, horizons) Dict
        +calculate_position_sizing(risk_metrics, portfolio) Dict
        +generate_stop_loss_take_profit(current, direction, risk) Dict
        +generate_comprehensive_signals(current, preds, risk, portfolio) Dict
    }

    class PredictionVisualizer {
        -Tuple figsize
        -Dict colors
        +plot_price_history_with_predictions(dates, hist, preds, ci, n_hist, title)
        +plot_training_history(histories, title)
        +plot_decomposition_results(components, dates, title)
        +plot_risk_analysis(report, prices, preds, title)
        +plot_trading_signals_dashboard(signals, current, preds, title)
        +plot_model_comparison(results, title)
    }

    class OilPricePredictionApp {
        -PredictionEngine engine
        -RiskAnalyzer risk_analyzer
        -TradingSignalGenerator signal_generator
        -PredictionVisualizer visualizer
        +run_basic_prediction(symbol, days, plot) Dict
        +run_advanced_analysis(symbol, config) Dict
        +run_comparison_analysis(symbols, wavelets) Dict
        +save_results(path) void
        +load_results(path) void
    }

    PredictionEngine *-- DataProcessor : composes
    PredictionEngine *-- WaveletAnalyzer : composes
    PredictionEngine *-- ModelBuilder : composes
    OilPricePredictionApp *-- PredictionEngine : uses
    OilPricePredictionApp *-- RiskAnalyzer : uses
    OilPricePredictionApp *-- TradingSignalGenerator : uses
    OilPricePredictionApp *-- PredictionVisualizer : uses
    TradingSignalGenerator ..> RiskAnalyzer : depends
```

## 2. 클래스 다이어그램 — API 레이어 (Pydantic 스키마)

```mermaid
classDiagram
    class PredictRequest {
        +str symbol
        +int days
        +str wavelet
        +int decomposition_level
        +int sequence_length
        +int epochs
    }

    class ComponentPrediction {
        +str name
        +List~float~ values
    }

    class PredictResponse {
        +str symbol
        +float current_price
        +List~str~ historical_dates
        +List~float~ historical_prices
        +List~float~ predictions
        +List~ComponentPrediction~ component_predictions
        +str wavelet
        +int decomposition_level
        +datetime generated_at
    }

    class HealthResponse {
        +str status
        +str tensorflow_version
        +bool gpu_available
    }

    class WaveletListResponse {
        +Dict~str, List~ wavelets
    }

    class PredictionService {
        <<module>>
        +run_prediction(req: PredictRequest) PredictResponse
        +list_wavelets() Dict
    }

    class PredictionsRouter {
        <<router>>
        +GET /api/wavelets
        +POST /api/predict
    }

    class HealthRouter {
        <<router>>
        +GET /api/health
    }

    PredictResponse *-- ComponentPrediction : contains
    PredictionsRouter ..> PredictRequest : accepts
    PredictionsRouter ..> PredictResponse : returns
    PredictionsRouter ..> WaveletListResponse : returns
    PredictionsRouter ..> PredictionService : delegates
    HealthRouter ..> HealthResponse : returns
    PredictionService ..> PredictionEngine : orchestrates
```

## 3. 컴포넌트 다이어그램 (Frontend ↔ Backend)

```mermaid
flowchart LR
    subgraph Frontend["Frontend (Vue 3 SPA)"]
        direction TB
        AV["AnalyzeView.vue<br/>입력 폼"]
        DV["DashboardView.vue<br/>결과 표시"]
        PC["PriceChart.vue<br/>Chart.js"]
        PS["Pinia Store<br/>usePredictionStore"]
        AC["api/client.ts<br/>axios instance"]
        AV --> PS
        DV --> PS
        DV --> PC
        PS --> AC
    end

    subgraph Backend["Backend (FastAPI)"]
        direction TB
        subgraph API["API Layer"]
            RH["routers/health.py"]
            RP["routers/predictions.py"]
            SV["services/<br/>prediction_service.py"]
            SC["schemas.py<br/>(Pydantic)"]
        end
        subgraph ML["ML Core"]
            PE["PredictionEngine"]
            DP["DataProcessor"]
            WA["WaveletAnalyzer"]
            MB["ModelBuilder"]
            RA["RiskAnalyzer"]
        end
        RP --> SV
        SV --> PE
        PE --> DP
        PE --> WA
        PE --> MB
    end

    subgraph External["External"]
        YF[("yfinance<br/>Yahoo Finance")]
        H5[("*.h5<br/>model checkpoints")]
    end

    AC -- "POST /api/predict<br/>GET /api/health" --> RP
    AC -- "GET /api/health" --> RH
    DP -- fetch --> YF
    MB -- save/load --> H5
```

## 4. 시퀀스 다이어그램 — 예측 요청 전체 흐름

```mermaid
sequenceDiagram
    actor User
    participant AV as AnalyzeView
    participant Store as Pinia Store
    participant Client as api/client.ts
    participant API as predictions router
    participant Svc as prediction_service
    participant Eng as PredictionEngine
    participant DP as DataProcessor
    participant WA as WaveletAnalyzer
    participant MB as ModelBuilder

    User->>AV: 파라미터 입력 + 제출
    AV->>Store: runPrediction(form)
    Store->>Client: predict(req)
    Client->>API: POST /api/predict
    API->>Svc: run_prediction(req)
    Svc->>Eng: new PredictionEngine(...)
    Svc->>Eng: run_full_pipeline(symbol, days, epochs)

    Eng->>DP: fetch_oil_data(symbol)
    DP-->>Eng: OHLCV DataFrame

    Eng->>WA: decompose(oil_prices)
    WA-->>Eng: {trend, detail_1..5}

    loop 각 컴포넌트
        Eng->>DP: create_sequences(component)
        DP-->>Eng: X, y, scaler
        Eng->>MB: create_model_by_type(type)
        MB-->>Eng: Keras Model
        Eng->>MB: train_model(X_tr, y_tr, X_val, y_val)
        MB-->>Eng: History
    end

    loop n_steps
        Eng->>Eng: predict_next_values(step)
    end
    Eng->>Eng: reconstruct_predictions()
    Eng-->>Svc: results Dict

    Svc-->>API: PredictResponse
    API-->>Client: 200 OK + JSON
    Client-->>Store: PredictResponse
    Store-->>AV: result 업데이트
    AV->>User: router.push('/') → Dashboard
```

## 5. 활동 다이어그램 — 학습·예측 파이프라인

```mermaid
flowchart TD
    Start([Start]) --> Fetch[Fetch OHLCV<br/>via yfinance]
    Fetch -->|성공| Prices[oil_prices 1D]
    Fetch -->|실패| Synth[Synthetic data 생성]
    Synth --> Prices
    Prices --> Decomp[Wavelet 분해<br/>level=5]
    Decomp --> Comp{{컴포넌트:<br/>trend + detail_1..5}}

    Comp --> Split[컴포넌트별<br/>Scaler + Sequence 생성]
    Split --> TrainTest[Sequential split<br/>80/10/10]
    TrainTest --> LoopTrain{{각 컴포넌트}}
    LoopTrain --> Build[config에 맞춰 모델 생성<br/>simple / bi / cnn_lstm / attn / ens]
    Build --> Train[train_model<br/>EarlyStop + ReduceLR]
    Train --> Store[component_models<br/>component_scalers 저장]

    Store --> LoopPred{{n_steps 반복 예측}}
    LoopPred --> PredStep[각 컴포넌트<br/>predict_next_values]
    PredStep --> InvScale[inverse_transform]
    InvScale --> Recon[Σ components<br/>reconstruct_predictions]

    Recon --> Risk[RiskAnalyzer<br/>generate_risk_report]
    Risk --> Signals[TradingSignalGenerator<br/>generate_comprehensive_signals]
    Signals --> Build2[PredictResponse 빌드]
    Build2 --> End([Return JSON])
```

## 6. 상태 다이어그램 — 프런트엔드 Prediction Store

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Loading: runPrediction(req)
    Loading --> Success: HTTP 200
    Loading --> Error: HTTP 4xx/5xx / timeout
    Success --> Idle: reset()
    Error --> Idle: reset()
    Success --> Loading: 재요청
    Error --> Loading: 재시도
```

## 7. 배포 다이어그램

```mermaid
flowchart TB
    subgraph Dev["Developer Machine"]
        subgraph Browser["Browser :5173"]
            SPA["Vue 3 SPA<br/>Vite dev server"]
        end
        subgraph Python["Python Process :8000"]
            UV["Uvicorn"]
            FA["FastAPI app"]
            TF["TensorFlow 2.x<br/>GPU optional"]
            UV --> FA
            FA --> TF
        end
        FS[("backend/<br/>*.h5 checkpoints")]
        TF -. "저장/로드" .-> FS
    end

    SPA -- "HTTP/JSON" --> UV
    TF -- "yfinance HTTPS" --> YF[("Yahoo Finance")]
```

## 8. 모델 계층 구조 (ML 아키텍처)

```mermaid
flowchart LR
    subgraph Ensemble["create_advanced_ensemble (Trend 전용)"]
        direction TB
        E1[Deep LSTM]
        E2[Wide + Residual LSTM]
        E3[Attention LSTM]
        G[Gating Network]
        E1 --> W[Weighted Sum]
        E2 --> W
        E3 --> W
        G -->|learned weights| W
    end

    subgraph Variants["컴포넌트별 모델 선택지"]
        direction LR
        S[Simple LSTM<br/>2× LSTM + Dense]
        B[Bidirectional LSTM<br/>2× BiLSTM + Dense]
        C[CNN-LSTM<br/>Conv1D + MaxPool + LSTM]
        A[Attention LSTM<br/>Scaled dot-product]
    end

    config["config_example.json<br/>model_config"] --> Factory[create_model_by_type]
    Factory --> Ensemble
    Factory --> Variants
```

---

## 참고

- Mermaid 문법: https://mermaid.js.org/
- 본 다이어그램은 현재 코드베이스(`backend/` + `frontend/src/`) 기준이며, 파일 구조가 바뀌면 함께 갱신해야 합니다.
- 아키텍처 세부사항은 [`architecture.md`](./architecture.md) 참조.
