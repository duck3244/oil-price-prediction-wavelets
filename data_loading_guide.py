#!/usr/bin/env python3
"""
GitHub 저장소 데이터로 모델 학습하기
ankit-maverick/Wavelets_course_project 데이터 활용 가이드
"""

import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime
import matplotlib.pyplot as plt

# 우리가 구축한 모듈들 import
from data_processor import DataProcessor
from wavelet_analyzer import WaveletAnalyzer
from model_builder import ModelBuilder
from predictor_engine import PredictionEngine
from risk_analyzer import RiskAnalyzer, TradingSignalGenerator
from visualization_tools import PredictionVisualizer, create_summary_report


class GitHubDataLoader:
    """
    GitHub 저장소에서 데이터를 로드하고 처리하는 클래스
    """

    def __init__(self,
                 repo_url="https://raw.githubusercontent.com/ankit-maverick/Wavelets_course_project/master/data/"):
        """
        GitHub 데이터 로더 초기화

        Args:
            repo_url: GitHub raw 데이터 URL 베이스
        """
        self.repo_url = repo_url
        self.data = None

    def download_csv_from_github(self, filename):
        """
        GitHub에서 CSV 파일 다운로드

        Args:
            filename: 다운로드할 파일명

        Returns:
            pandas.DataFrame: 로드된 데이터
        """
        try:
            # GitHub raw URL 구성
            file_url = f"{self.repo_url}{filename}"
            print(f"Downloading data from: {file_url}")

            # 파일 다운로드
            response = requests.get(file_url)
            response.raise_for_status()  # HTTP 에러 체크

            # CSV 데이터 파싱
            data = pd.read_csv(io.StringIO(response.text))

            print(f"Successfully loaded {filename}")
            print(f"Data shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")

            return data

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return None
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def load_oil_price_data(self, filename="oil_prices.csv"):
        """
        유가 데이터 로드 및 전처리

        Args:
            filename: 유가 데이터 파일명

        Returns:
            tuple: (dates, prices) 또는 None
        """
        # 가능한 파일명들 시도
        possible_filenames = [
            filename,
            "oil_prices.csv",
            "crude_oil_price.csv",
            "oil_data.csv",
            "wti_prices.csv",
            "brent_prices.csv",
            "price_data.csv",
            "data.csv"
        ]

        for fname in possible_filenames:
            print(f"\nTrying to load: {fname}")
            data = self.download_csv_from_github(fname)

            if data is not None:
                # 데이터 구조 분석 및 전처리
                processed_data = self._preprocess_oil_data(data, fname)
                if processed_data is not None:
                    return processed_data

        print("Failed to load any oil price data from GitHub")
        return None

    def _preprocess_oil_data(self, data, filename):
        """
        유가 데이터 전처리

        Args:
            data: 원본 데이터프레임
            filename: 파일명

        Returns:
            tuple: (dates, prices) 또는 None
        """
        try:
            print(f"\nPreprocessing {filename}...")
            print(f"Original columns: {list(data.columns)}")
            print(f"First few rows:\n{data.head()}")

            # 컬럼이 모두 숫자인 경우 (날짜 컬럼이 없는 경우)
            if all(str(col).replace('.', '').replace('-', '').isdigit() for col in data.columns):
                print("⚠️ No date column detected. Assuming data is time series without dates.")

                # 가장 적절한 가격 컬럼 선택 (보통 마지막 컬럼이 종가)
                price_col_idx = len(data.columns) - 1  # 마지막 컬럼
                prices = data.iloc[:, price_col_idx].values

                # 날짜 인덱스 생성 (일일 데이터라고 가정)
                dates = pd.date_range(start='2000-01-01', periods=len(prices), freq='D')

                print(f"Generated date range: {dates[0]} to {dates[-1]}")
                print(f"Using column {price_col_idx} as price data")

            else:
                # 기존 로직 (날짜 컬럼이 있는 경우)
                date_columns = ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp', 'Timestamp']
                price_columns = ['price', 'Price', 'close', 'Close', 'value', 'Value', 'oil_price', 'crude_price']

                # 날짜 컬럼 찾기
                date_col = None
                for col in date_columns:
                    if col in data.columns:
                        date_col = col
                        break

                # 가격 컬럼 찾기
                price_col = None
                for col in price_columns:
                    if col in data.columns:
                        price_col = col
                        break

                # 컬럼이 명확하지 않은 경우 자동 추정
                if date_col is None:
                    # 첫 번째 컬럼이 날짜일 가능성이 높음
                    date_col = data.columns[0]
                    print(f"Auto-detected date column: {date_col}")

                if price_col is None:
                    # 숫자형 컬럼 중 마지막이 가격일 가능성이 높음
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        price_col = numeric_cols[-1]  # 보통 마지막 컬럼이 종가
                        print(f"Auto-detected price column: {price_col}")
                    else:
                        print("No numeric columns found for price")
                        return None

                # 날짜 변환
                try:
                    dates = pd.to_datetime(data[date_col])
                except:
                    print(f"Failed to parse dates in column: {date_col}")
                    # 인덱스가 날짜인지 확인
                    try:
                        dates = pd.to_datetime(data.index)
                    except:
                        print("Failed to parse dates from index as well")
                        return None

                # 가격 데이터 추출
                try:
                    prices = pd.to_numeric(data[price_col], errors='coerce')
                    # NaN 값 제거
                    valid_idx = ~prices.isna()
                    dates = dates[valid_idx]
                    prices = prices[valid_idx].values
                except:
                    print(f"Failed to parse prices in column: {price_col}")
                    return None

            # 데이터 품질 확인
            if len(prices) < 100:
                print(f"Warning: Only {len(prices)} data points available. Minimum 100 recommended.")

            # NaN 및 무한값 제거
            valid_mask = np.isfinite(prices)
            dates = dates[valid_mask]
            prices = prices[valid_mask]

            if len(prices) == 0:
                print("Error: No valid price data after cleaning")
                return None

            # 데이터 정렬 (날짜 순)
            if len(dates) == len(prices):
                sort_idx = np.argsort(dates)
                dates = dates[sort_idx]
                prices = prices[sort_idx]

            print(f"Processed data successfully:")
            print(f"  Date range: {dates[0]} to {dates[-1]}")
            print(f"  Data points: {len(prices)}")
            print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")

            return dates, prices

        except Exception as e:
            print(f"Error preprocessing data: {e}")
            import traceback
            traceback.print_exc()
            return None


def train_with_github_data(github_data_path=None):
    """
    GitHub 저장소의 데이터로 모델 훈련

    Args:
        github_data_path: GitHub 데이터 파일 경로 (선택사항)

    Returns:
        dict: 훈련 결과
    """
    print("🔄 Starting training with GitHub repository data...")
    print("=" * 60)

    # 1. GitHub 데이터 로더 초기화
    loader = GitHubDataLoader()

    # 2. 데이터 로드
    if github_data_path:
        data_result = loader.load_oil_price_data(github_data_path)
    else:
        data_result = loader.load_oil_price_data()

    if data_result is None:
        print("❌ Failed to load data from GitHub repository")
        print("💡 Falling back to Yahoo Finance data...")
        return train_with_yahoo_finance()

    dates, prices = data_result

    # 3. 데이터를 우리 시스템에 맞게 설정
    engine = PredictionEngine(
        wavelet='db4',
        decomposition_level=5,
        sequence_length=60,
        prediction_horizon=1
    )

    # 데이터를 엔진에 직접 설정
    engine.data_processor.oil_prices = prices
    engine.data_processor.dates = dates
    engine.data_processor.raw_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices * (1 + np.random.normal(0, 0.005, len(prices))),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
        'Volume': np.random.randint(1000000, 5000000, len(prices))
    }).set_index('Date')

    print(f"✅ Data loaded successfully:")
    print(f"   Data points: {len(prices)}")
    print(f"   Date range: {dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}")
    print(f"   Current price: ${prices[-1]:.2f}")

    # 4. 웨이블릿 분해
    print(f"\n🌊 Performing wavelet decomposition...")
    components = engine.wavelet_decomposition(prices, plot=True)

    # 5. 모델 훈련
    print(f"\n🚀 Training models...")
    training_results = engine.train_component_models(
        epochs=100,
        batch_size=32,
        plot_results=True
    )

    # 6. 예측 생성
    if engine.is_trained:
        print(f"\n🔮 Generating predictions...")
        prediction_results = engine.predict(30)  # 30일 예측

        # 7. 위험 분석
        print(f"\n⚠️ Performing risk analysis...")
        risk_analyzer = RiskAnalyzer()
        risk_report = risk_analyzer.generate_risk_report(
            prices, prediction_results['predictions'],
            prediction_results['component_predictions']
        )

        # 8. 트레이딩 신호
        print(f"\n📈 Generating trading signals...")
        signal_generator = TradingSignalGenerator()
        trading_signals = signal_generator.generate_comprehensive_signals(
            prediction_results['current_price'],
            prediction_results['predictions'],
            risk_report
        )

        # 9. 시각화
        print(f"\n📊 Creating visualizations...")
        visualizer = PredictionVisualizer()

        # 가격 예측 차트
        visualizer.plot_price_history_with_predictions(
            dates, prices, prediction_results['predictions'],
            title="Oil Price Prediction - GitHub Data"
        )

        # 위험 분석 대시보드
        visualizer.plot_risk_analysis_dashboard(
            risk_report, prices, prediction_results['predictions']
        )

        # 트레이딩 신호 대시보드
        visualizer.plot_trading_signals_dashboard(
            trading_signals, prediction_results['current_price'],
            prediction_results['predictions']
        )

        # 종합 리포트
        create_summary_report(prediction_results, risk_report, trading_signals)

        # 10. 결과 요약
        print(f"\n📋 TRAINING RESULTS SUMMARY")
        print("=" * 50)
        print(f"✅ Successfully trained {len(engine.component_models)} models")
        print(f"📊 Used {len(prices)} data points")
        print(f"🔮 Generated {len(prediction_results['predictions'])} predictions")
        print(f"⚠️ Risk Level: {risk_report.get('overall_risk', 'Unknown')}")
        print(f"📈 Primary Signal: {trading_signals.get('primary_signal', {}).get('signal', 'Unknown')}")

        return {
            'data_source': 'GitHub Repository',
            'data_points': len(prices),
            'models_trained': len(engine.component_models),
            'prediction_results': prediction_results,
            'risk_report': risk_report,
            'trading_signals': trading_signals,
            'training_results': training_results
        }

    else:
        print("❌ Model training failed")
        return None


def train_with_yahoo_finance():
    """
    Yahoo Finance 데이터로 대체 훈련
    """
    print("🔄 Using Yahoo Finance as fallback data source...")

    engine = PredictionEngine()
    results = engine.run_full_pipeline(
        symbol="CL=F",
        start_date="2015-01-01",
        n_predictions=30,
        plot_decomposition=True
    )

    return results


def train_with_custom_csv(csv_file_path):
    """
    로컬 CSV 파일로 훈련

    Args:
        csv_file_path: CSV 파일 경로

    Returns:
        dict: 훈련 결과
    """
    try:
        print(f"🔄 Loading data from local file: {csv_file_path}")

        # CSV 파일 로드
        data = pd.read_csv(csv_file_path)
        print(f"Loaded CSV with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")

        # GitHub 데이터 로더의 전처리 함수 재사용
        loader = GitHubDataLoader()
        processed_data = loader._preprocess_oil_data(data, csv_file_path)

        if processed_data is None:
            print("❌ Failed to preprocess CSV data")
            return None

        dates, prices = processed_data

        # 나머지는 GitHub 데이터 훈련과 동일
        return train_with_processed_data(dates, prices, f"Local CSV: {csv_file_path}")

    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return None


def train_with_processed_data(dates, prices, data_source="Custom Data"):
    """
    전처리된 데이터로 훈련 (공통 함수)
    """
    print(f"\n🚀 Starting training with {data_source}...")

    engine = PredictionEngine(
        wavelet='db4',
        decomposition_level=4,  # 데이터가 적을 수 있으므로 줄임
        sequence_length=30,  # 시퀀스 길이도 줄임
        prediction_horizon=1
    )

    # 데이터 설정
    engine.data_processor.oil_prices = prices
    engine.data_processor.dates = dates
    engine.data_processor.raw_data = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.normal(0, 0.005, len(prices))),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
        'Volume': np.random.randint(1000000, 5000000, len(prices))
    }, index=dates)

    print(f"✅ Data configured:")
    print(f"   Data points: {len(prices)}")
    print(f"   Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")

    # 웨이블릿 분해 (wavelet_analyzer 직접 사용)
    print(f"\n🌊 Performing wavelet decomposition...")
    try:
        components = engine.wavelet_analyzer.decompose(prices, plot=True)
        print(f"✅ Wavelet decomposition completed: {len(components)} components")
        engine.components = components
    except Exception as e:
        print(f"❌ Wavelet decomposition failed: {e}")
        return None

    # 모델 훈련
    print(f"\n🤖 Training component models...")
    try:
        # train_component_models 메서드의 올바른 인수들 사용
        model_config = {
            'trend': 'ensemble',
            'detail_1': 'bidirectional',
            'detail_2': 'cnn_lstm',
            'detail_3': 'simple',
            'detail_4': 'simple'
        }

        training_config = {
            'epochs': 50,
            'batch_size': 16,
            'train_ratio': 0.8,
            'validation_ratio': 0.1
        }

        training_results = engine.train_component_models(
            model_config=model_config,
            **training_config
        )

        if not engine.is_trained:
            print("❌ No models were trained successfully")
            return None

        print(f"✅ Successfully trained {len(engine.component_models)} models")

        # 훈련 히스토리 시각화 (별도로 실행)
        try:
            if hasattr(engine, 'component_histories') and engine.component_histories:
                print(f"📊 Creating training history plots...")
                visualizer = PredictionVisualizer()
                visualizer.plot_training_history(engine.component_histories)
        except Exception as e:
            print(f"Warning: Training history visualization failed: {e}")

    except Exception as e:
        print(f"❌ Model training failed: {e}")
        print(f"Available methods in engine: {[method for method in dir(engine) if not method.startswith('_')]}")
        import traceback
        traceback.print_exc()
        return None

    # 예측 생성
    print(f"\n🔮 Generating predictions...")
    try:
        prediction_results = engine.predict(10)  # 10일 예측

        # 위험 분석
        print(f"\n⚠️ Performing risk analysis...")
        risk_analyzer = RiskAnalyzer()
        risk_report = risk_analyzer.generate_risk_report(
            prices, prediction_results['predictions'],
            prediction_results['component_predictions']
        )

        # 트레이딩 신호
        print(f"\n📈 Generating trading signals...")
        signal_generator = TradingSignalGenerator()
        trading_signals = signal_generator.generate_comprehensive_signals(
            prediction_results['current_price'],
            prediction_results['predictions'],
            risk_report
        )

        # 시각화
        print(f"\n📊 Creating visualizations...")
        visualizer = PredictionVisualizer()

        # 간단한 시각화만 생성 (에러 방지)
        try:
            visualizer.plot_price_history_with_predictions(
                dates, prices, prediction_results['predictions'],
                title=f"Oil Price Prediction - {data_source}"
            )
        except Exception as e:
            print(f"Warning: Visualization error: {e}")

        # 결과 요약
        print(f"\n📋 TRAINING RESULTS SUMMARY")
        print("=" * 50)
        print(f"✅ Data source: {data_source}")
        print(f"📊 Data points used: {len(prices)}")
        print(f"🤖 Models trained: {len(engine.component_models)}")
        print(f"🔮 Predictions generated: {len(prediction_results['predictions'])}")
        print(f"💰 Current price: ${prediction_results['current_price']:.2f}")
        print(f"📈 Next day prediction: ${prediction_results['predictions'][0]:.2f}")
        print(f"⚠️ Risk level: {risk_report.get('overall_risk', 'Unknown')}")

        return {
            'data_source': data_source,
            'data_points': len(prices),
            'models_trained': len(engine.component_models),
            'prediction_results': prediction_results,
            'risk_report': risk_report,
            'trading_signals': trading_signals,
            'training_results': training_results
        }

    except Exception as e:
        print(f"❌ Prediction generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# 메인 실행 함수들
def main_github_training():
    """GitHub 데이터로 메인 훈련 실행"""
    print("🛢️ OIL PRICE PREDICTION WITH GITHUB DATA")
    print("=" * 60)

    # 방법 1: 자동으로 GitHub 저장소에서 데이터 찾기
    results = train_with_github_data()

    if results:
        print("\n🎉 Training completed successfully!")
        return results
    else:
        print("\n❌ Training failed")
        return None


def main_custom_csv():
    """사용자 CSV 파일로 훈련"""
    csv_path = input("Enter CSV file path: ")
    return train_with_custom_csv(csv_path)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train oil price prediction with various data sources")
    parser.add_argument('csv_file', nargs='?', help='CSV file path (positional argument)')
    parser.add_argument('--data-file', help='Specific data file name in GitHub repo')
    parser.add_argument('--csv-file', help='Local CSV file path (alternative to positional)')
    parser.add_argument('--mode', choices=['github', 'csv', 'yahoo'], default='auto',
                        help='Data source mode (auto, github, csv, yahoo)')

    args = parser.parse_args()

    # CSV 파일 경로 결정 (positional argument 우선)
    csv_file_path = args.csv_file or args.csv_file

    # 모드 자동 결정
    if args.mode == 'auto':
        if csv_file_path and csv_file_path.endswith('.csv'):
            mode = 'csv'
        elif args.data_file:
            mode = 'github'
        else:
            mode = 'github'  # 기본값
    else:
        mode = args.mode

    print(f"🔄 Running in {mode} mode")

    try:
        if mode == 'github':
            print("📊 Using GitHub repository data")
            results = train_with_github_data(args.data_file)
        elif mode == 'csv':
            if csv_file_path:
                print(f"📁 Using local CSV file: {csv_file_path}")
                results = train_with_custom_csv(csv_file_path)
            else:
                print("❌ CSV mode selected but no file specified")
                print("Usage: python data_loading_guide.py data/data.csv")
                print("   or: python data_loading_guide.py --csv-file data/data.csv")
                sys.exit(1)
        elif mode == 'yahoo':
            print("📈 Using Yahoo Finance data")
            results = train_with_yahoo_finance()

        if results:
            print(f"\n✅ Training completed successfully!")
            print(f"📊 Data source: {results.get('data_source', 'Unknown')}")
            if 'models_trained' in results:
                print(f"🤖 Models trained: {results['models_trained']}")
            if 'data_points' in results:
                print(f"📈 Data points used: {results['data_points']}")
        else:
            print(f"\n❌ Training failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)