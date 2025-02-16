# GPT_binance_o3_mini
GPT-o3-mini 모델로 4시간마다 거래 결정을 받아오고 이를 바탕으로 롱,숏 포지션을 잡는 자동매매 프로그램입니다.

```markdown
# Bitcoin Trading Bot

이 프로젝트는 Binance 선물 거래 API와 OpenAI의 챗 API를 활용하여 자동 매매 전략을 구현한 Python 기반의 트레이딩 봇입니다. 기술적 지표 분석(ta 라이브러리 사용), 계좌 및 포지션 모니터링, AI 기반 거래 결정 생성, 주문 실행 및 거래 기록 저장(SQLite) 등의 기능을 제공합니다.

---

## 주요 기능

- **시장 데이터 수집 및 기술적 지표 계산:**  
  Binance API를 통해 4시간, 1시간, 1일 간격의 시장 데이터를 가져오고, Bollinger Bands, RSI, MACD, 이동평균선, ATR 등 다양한 기술적 지표를 계산합니다.

- **계좌 및 포지션 상태 확인:**  
  선물 계좌 잔고와 롱/숏 포지션 정보를 조회하여 현재 자산 상태를 파악합니다.

- **AI 기반 거래 결정:**  
  OpenAI 챗 API를 사용하여 최근 거래 기록과 현재 시장 데이터를 바탕으로 거래 결정을 생성합니다. 거래 결정은 롱, 숏 또는 홀드 중 하나로 결정되며, 포지션 비율 및 사유도 함께 제공합니다.

- **거래 실행 및 기록:**  
  기존 포지션을 청산한 후 새로운 주문을 실행하고, 결과를 SQLite 데이터베이스에 저장합니다.

- **스케줄링:**  
  지정된 시간(`SCHEDULE_TIMES`)에 자동으로 거래 작업을 실행할 수 있습니다.

- **디스코드 로깅 (옵션):**  
  Discord 웹훅을 통해 로그 메시지를 전송할 수 있으며, 이를 통해 실시간 알림을 받을 수 있습니다.

---

## 요구 사항

- **Python:** 3.7 이상
- **필수 라이브러리:**  
  - pandas
  - requests
  - schedule
  - ta
  - python-dotenv
  - pydantic
  - openai
  - python-binance

### 설치 예시

```bash
pip install pandas requests schedule ta python-dotenv pydantic openai python-binance
```

---

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래와 같이 API 키 및 설정 값을 입력하세요.

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
OPENAI_API_KEY=your_openai_api_key
DISCORD_WEBHOOK_URL=your_discord_webhook_url  # 선택 사항 (없으면 비워두세요)
SERPAPI_API_KEY=your_serpapi_api_key           # 선택 사항 (미사용 시 비워두세요)
```

---

## 구성 및 파일 설명

- **설정 및 라이브러리 임포트:**  
  `dotenv`를 사용해 환경 변수를 로드하며, 거래 심볼, 레버리지, 최소 주문 수량, 시간대별 데이터 설정, 기술적 지표 설정 등을 정의합니다.

- **로깅 설정:**  
  `RotatingFileHandler`와 콘솔 핸들러를 사용하여 로그를 파일과 콘솔에 기록하며, 옵션으로 Discord 웹훅을 통한 로그 전송 기능도 포함되어 있습니다.

- **모델 정의:**  
  Pydantic을 사용하여 거래 결정(`DualTradingDecision`), 포지션(`Position`, `Positions`), 거래 로그(`TradeLog`) 등의 데이터 모델을 정의합니다.

- **헬퍼 함수:**  
  주문 수량 조정, 포지션 가치 및 포트폴리오 수익률 계산, 거래 전략 파일 읽기 등의 공통 유틸리티를 제공합니다.

- **기술적 지표 추가:**  
  `ta` 라이브러리를 활용하여 DataFrame에 다양한 기술적 지표를 추가하는 함수(`add_indicators`)를 구현하였습니다.

- **서비스 클래스:**  
  - **BinanceService:** Binance API를 통해 잔고 조회, 포지션 정보, 시장 데이터 수집, 주문 실행 등의 기능 제공  
  - **MarketDataService:** 대체 API를 통해 시장 심리 지표(예: Fear & Greed Index) 조회  
  - **AIService:** OpenAI 챗 API를 사용하여 거래 반성과 결정 생성 및 로그 기록  
  - **DatabaseService:** SQLite를 사용하여 거래 기록 저장 및 조회

- **트레이딩 전략:**  
  `TradingStrategy` 클래스는 계좌 상태 및 시장 데이터를 수집하고, AI 기반 거래 결정을 생성하여 주문을 실행하고 결과를 로그 및 데이터베이스에 기록합니다.

- **메인 실행부:**  
  명령줄 인자(`--test`)에 따라 즉시 실행 모드와 스케줄 모드를 지원합니다. 스케줄 모드에서는 설정된 시간(`SCHEDULE_TIMES`)에 자동으로 거래 작업이 실행됩니다.

---

## 사용 방법

### 즉시 실행 모드 (테스트)

테스트 및 즉시 실행을 위해 터미널에서 아래 명령을 실행합니다:
MacOS의 경우 python 대신 python3 사용합니다.

```bash
python trading.py --test
```

이 모드에서는 한 번 거래 작업을 실행합니다.

### 스케줄 모드

자동 실행을 원한다면 인자 없이 실행합니다:

```bash
python trading.py
```

이 경우, 설정된 스케줄(`SCHEDULE_TIMES`)에 따라 매일 정해진 시간에 거래 작업이 자동으로 수행됩니다.

---


## 추가 설정

- **거래 전략 파일:**  
  프로젝트 루트에 `strategy.txt` 파일을 생성하고, 거래 전략을 기술하세요. 이 파일의 내용은 AI가 거래 결정을 생성할 때 참고됩니다.

- **디스코드 로그 전송:**  
  Discord 웹훅 URL이 설정된 경우, 로그 메시지가 Discord 채널로 전송됩니다. 이 기능을 활성화하려면 `.env` 파일에 `DISCORD_WEBHOOK_URL` 값을 설정하세요.

- **로그 파일:**  
  로그는 `trading_bot.log` 파일과 `logs/gpt/` 디렉터리에 저장됩니다. 파일 관리에 유의하세요.

---

## 주의 사항

- **API 호출 및 예외 처리:**  
  네트워크 문제, API 제한 등에 따른 예외가 발생할 수 있으므로, 추가적인 예외 처리 및 재시도 로직이 필요할 수 있습니다.

- **보안:**  
  API 키 및 민감 정보는 `.env` 파일을 통해 관리하며, 코드 저장소에 포함되지 않도록 주의하세요.

- **동시성:**  
  `trading_in_progress` 변수로 중복 실행을 방지하고 있으나, 멀티스레딩 환경에서는 추가적인 동기화가 필요할 수 있습니다.

---

## 라이선스

이 프로젝트의 라이선스는 [MIT License](LICENSE)와 같습니다.

---

## 기여

버그 제보, 기능 개선 제안, 풀 리퀘스트 등 모든 기여를 환영합니다!

---

## 참고 자료

- [Binance API 문서](https://binance-docs.github.io/apidocs/futures/en/)
- [OpenAI API 문서](https://beta.openai.com/docs/)
- [ta 라이브러리](https://technical-analysis-library-in-python.readthedocs.io/)

```
