# ECG-Based Personal Identification Capstone Project

본 프로젝트는 ECG(Electrocardiogram, 심전도) 신호를 이용한 개인식별 및 인증 시스템 개발을 목표로 한다.

ECG는 개인마다 심장 구조와 전기적 전도 특성이 다르기 때문에 생체 특징으로 활용될 수 있다. 본 프로젝트에서는 ECG 신호를 전처리하고, beat 단위 segment를 생성한 뒤, 딥러닝 기반 특징 벡터를 추출하여 유사도 기반 개인식별 및 인증을 수행한다.

또한 Galaxy Watch 기반 ECG 측정 앱과 Flask 기반 웹 인증 시스템을 함께 구성하여, 웨어러블 기기에서 측정한 ECG 데이터를 서버로 전송하고 웹 대시보드에서 인증 결과를 확인할 수 있는 구조로 구현하였다.

## Project Overview

```text
Subject ID Selection
→ Watch ECG Measurement or ECG JSON Upload
→ Signal Quality Evaluation
→ R-Peak Detection
→ Beat Segmentation
→ ECG Embedding Generation
→ Template Registration or Similarity Verification
→ Authentication Result
→ Web Dashboard Visualization
→ Authentication Log and PDF Report
```

## Main Features

- ECG 원시 신호 로드 및 전처리
- R-peak 검출 기반 beat segmentation
- RR interval 기반 ECG segment 추출
- 학습용 ECG dataset 생성
- Plain CNN1D 기반 개인식별 모델 구조 구현
- ECG embedding vector 추출 구조 구현
- Cosine similarity 기반 인증 구조 구현
- Galaxy Watch ECG 측정 앱 구현
- ECG JSON 데이터 생성 및 서버 전송 구조 구현
- Flask 기반 ECG 인증 웹 시스템 구현
- Subject ID 기반 사용자 등록 및 인증 처리
- 수동 업로드 ECG 데이터의 현재 사용자 ID 기준 처리
- 등록 템플릿 생성 전 ECG 신호 품질 평가
- 파일 해시 기반 중복 ECG 등록 방지
- ECG 파형, 신호 품질, 유사도, 인증 결과 시각화
- 인증 기록 저장 및 PDF 리포트 생성

## Project Structure

```text
26-1_Capstone/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── dataset.py
│   ├── split.py
│   └── model.py
│
├── ECGMonitor/
│   └── Galaxy Watch ECG measurement application
│
└── ECG_Auth/
    ├── app.py
    ├── requirements.txt
    ├── docs/
    │   └── auth_policy.md
    ├── scripts/
    │   ├── calibrate_threshold.py
    │   └── check_runtime.py
    ├── services/
    ├── src/
    ├── templates/
    └── static/
```

## Source Code Description

| Path | Description |
|---|---|
| `src/` | ECG 전처리, segmentation, dataset 생성, 모델 구조 코드 |
| `ECGMonitor/` | Galaxy Watch ECG 측정 및 서버 전송용 Android 앱 |
| `ECG_Auth/app.py` | Flask 기반 ECG 인증 웹 서버 실행 파일 |
| `ECG_Auth/docs/auth_policy.md` | 사용자 ID 처리, 등록 템플릿 기준, threshold 정책 문서 |
| `ECG_Auth/scripts/calibrate_threshold.py` | ECG 데이터 분포 기반 similarity threshold 보정 스크립트 |
| `ECG_Auth/scripts/check_runtime.py` | TensorFlow, 모델 파일, config, DB 상태 점검 스크립트 |
| `ECG_Auth/services/` | ECG 데이터 처리, PDF 처리, Raw ECG 처리, 추론, HRV 분석 서비스 |
| `ECG_Auth/src/` | ECG 전처리, segmentation, embedding, evaluation 등 인증 파이프라인 코드 |
| `ECG_Auth/templates/index.html` | ECG 인증 웹 대시보드 화면 |
| `ECG_Auth/static/css/style.css` | 웹 대시보드 UI 스타일 |
| `ECG_Auth/static/js/app.js` | 파일 업로드, 서버 요청, 결과 표시 등 프론트엔드 로직 |

## Dataset

본 프로젝트는 ECG-ID Database 및 Galaxy Watch 기반 ECG 데이터를 활용할 수 있도록 설계하였다.

ECG 원본 데이터는 용량 및 개인정보 보호 문제로 GitHub 저장소에 포함하지 않는다. 사용자는 필요한 ECG 데이터를 직접 준비한 뒤, 각 모듈의 경로 설정에 맞게 배치해야 한다.

## Wearable ECG App

`ECGMonitor/`는 Galaxy Watch 기반 ECG 측정 앱이다.

```text
Galaxy Watch ECG Measurement
→ ECG Sample Collection
→ ECG JSON Generation
→ Local Save
→ Flask Server Upload
```

서버 주소는 `MainActivity.java`에서 설정한다.

```java
private static final String SERVER_URL =
        "http://YOUR_SERVER_IP:5000/api/ecg-json/receive";
```

실행 전 `YOUR_SERVER_IP`를 Flask 서버가 실행 중인 PC의 IP 주소로 변경해야 한다.

## Web Authentication System

`ECG_Auth/`는 Flask 기반 ECG 인증 웹 시스템이다.

```text
Subject ID Selection
→ ECG JSON Upload or Watch ECG Receive
→ Signal Quality Evaluation
→ R-Peak Detection
→ Beat Segment Extraction
→ ECG Embedding Generation
→ Template Registration or Similarity Verification
→ Authentication Result Display
```

웹 대시보드는 다음 정보를 표시한다.

- ECG 파형
- 신호 품질 등급
- 추정 심박수
- R-peak 수 및 RR 기반 지표
- Cosine similarity
- 일치율
- 인증 성공 또는 실패 결과
- 인증 기록 및 PDF 리포트

## Authentication Policy

본 시스템은 사용자를 이름이 아닌 Subject ID(`S001`, `S002` 등) 기준으로 구분한다.

수동 업로드된 ECG JSON 파일도 파일명이나 JSON 내부 사용자 정보보다 현재 웹 UI에서 선택된 Subject ID를 우선하여 처리한다. 따라서 등록 템플릿과 인증 기록은 선택된 Subject ID 기준으로 저장된다.

등록 템플릿은 이후 인증 비교의 기준이 되므로 인증 단계보다 더 엄격한 신호 품질 기준을 적용한다. 샘플 수, 측정 길이, collection completeness, lead-off, invalid sample, flat tail, RR 안정성 등을 기준으로 등록 가능 여부를 판단한다.

인증 단계에서는 현재 ECG embedding과 등록 템플릿 embedding의 cosine similarity를 비교한다. 기본 threshold는 `0.85`이며, 해당 기준값은 외부 표준이 아닌 프로젝트 실험 파라미터이다. 데이터 분포 실험을 통해 threshold를 보정할 수 있다.

동일한 ECG 원본 파일이 서로 다른 Subject ID에 중복 등록되는 것을 방지하기 위해 파일 해시 기반 중복 검사를 수행한다.

자세한 정책은 다음 문서를 참고한다.

```text
ECG_Auth/docs/auth_policy.md
```

## Requirements

Python 기반 ECG 처리 및 웹 서버 실행에 사용하는 주요 패키지는 다음과 같다.

```text
numpy
pandas
scipy
wfdb
neurokit2
tensorflow
flask
matplotlib
PyMuPDF
Pillow
```

일부 실험 코드에서는 PyTorch 기반 모델 구조를 함께 사용할 수 있다.

설치 명령어는 다음과 같다.

```bash
pip install -r requirements.txt
```

Flask 웹 시스템 실행 예시는 다음과 같다.

```bash
cd ECG_Auth
python app.py
```

실행 환경 점검은 다음 명령어로 확인할 수 있다.

```bash
cd ECG_Auth
python scripts/check_runtime.py
```

Android 앱은 Android Studio에서 실행한다.

Samsung Health Sensor SDK 사용 시 필요한 SDK 파일은 직접 다운로드하여 `ECGMonitor/app/libs/`에 추가해야 한다. SDK 바이너리 파일은 저장소에 포함하지 않는다.

## Threshold Calibration

기본 similarity threshold는 `0.85`이다.

해당 값은 프로젝트 기준값이며, 로컬 ECG 데이터 분포를 기반으로 보정할 수 있다.

```bash
cd ECG_Auth
python scripts/calibrate_threshold.py --write-config
```

보정 스크립트는 같은 사용자 ECG 쌍과 다른 사용자 ECG 쌍의 cosine similarity 분포를 비교하여 threshold 후보를 계산한다. 결과는 다음 경로에 저장된다.

```text
ECG_Auth/outputs/threshold_calibration.json
ECG_Auth/models/plain_cnn1d_config.json
```

## Repository Policy

다음 파일은 GitHub 저장소에 포함하지 않는다.

```text
ECG 원본 데이터
사용자 ECG JSON 파일
업로드된 PDF 파일
추출된 ECG 결과 파일
사용자별 등록 템플릿 embedding 파일
학습 모델 가중치
출력 이미지 및 실험 결과
Android build output
.gradle/
.idea/
local.properties
APK 파일
Samsung Health Sensor SDK .aar 파일
```

## Development Status

현재 포함된 구현 범위는 다음과 같다.

- ECG preprocessing
- R-peak detection
- Beat segmentation
- Dataset generation
- Dataset splitting
- Plain CNN1D model architecture
- ECG embedding extraction structure
- Cosine similarity-based authentication structure
- Galaxy Watch ECG monitor app
- ECG JSON generation
- ECG JSON server upload structure
- Flask-based ECG authentication web system
- Subject ID-based registration and verification flow
- Registration quality gate
- File hash-based duplicate ECG registration check
- Structured ECG quality failure reason
- Web dashboard UI
- ECG waveform and authentication result visualization
- Authentication log and PDF report generation

## Notes

본 프로젝트는 ECG 기반 개인식별 및 인증 시스템의 캡스톤 설계 결과물이다.

ECG 신호 품질 정보와 인증 결과는 개인식별 및 인증 연구 목적의 보조 지표이며, 질병 진단 또는 의료 판단 목적으로 사용하지 않는다.
