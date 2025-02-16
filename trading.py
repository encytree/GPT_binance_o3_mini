# -------------------------------
# IMPORTS
# -------------------------------
import os
import math
import time
import json
import logging
import sqlite3
import argparse
import requests
import schedule
import pandas as pd
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional, Union
import ta
from dotenv import load_dotenv
from openai import OpenAI
from binance.client import Client
from binance.enums import *
from logging.handlers import RotatingFileHandler

# -------------------------------
# SETTINGS
# -------------------------------
load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

TRADING_SYMBOL = 'BTCUSDT'
DEFAULT_LEVERAGE = 1
MIN_QUANTITY = 0.001
QUANTITY_STEP = 0.001

DATABASE_NAME = 'bitcoin_trades.db'
SCHEDULE_TIMES = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']

TIMEFRAMES = {
    '4h': {'interval': 'KLINE_INTERVAL_4HOUR', 'limit': 300},
    '1h': {'interval': 'KLINE_INTERVAL_1HOUR', 'limit': 300},
    '1d': {'interval': 'KLINE_INTERVAL_1DAY', 'limit': 300}
}

INDICATOR_SETTINGS = {
    'bollinger_bands': {'window': 20, 'window_dev': 2},
    'rsi': {'window': 14},
    'macd': {'window_slow': 26, 'window_fast': 12, 'window_sign': 9},
    'moving_averages': {
        'sma_window': 20,
        'ema_window': 12,
        'ema_50_window': 50,
        'ema_200_window': 200
    },
    'atr': {'window': 14},
    'cci': {'window': 20},
    'mfi': {'window': 14},
    'standard_deviation': {'window': 20},
    'vwap': {'window': 14},
    'cmf': {'window': 20},
    'stoch_rsi': {'window': 14, 'smooth_window': 3},
    'williams_r': {'window': 14},
    'adx': {'window': 14},
    'ichimoku': {'window1': 9, 'window2': 26, 'window3': 52},
    'dmi': {'window': 14}
}

# -------------------------------
# LOGGING CONFIG
# -------------------------------
class DiscordHandler(logging.Handler):
    def __init__(self, webhook_url):
        super().__init__()
        self.webhook_url = webhook_url

    def emit(self, record):
        try:
            log_entry = self.format(record)
            data = {"content": log_entry}
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.webhook_url, data=json.dumps(data), headers=headers)
            # Discord 웹훅은 성공 시 보통 204 No Content 응답을 반환합니다.
            if response.status_code != 204:
                logging.error(f"Failed to send log to Discord webhook: {response.status_code} {response.text}")
        except Exception as e:
            logging.error(f"Exception in DiscordHandler.emit: {e}")

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    file_handler = RotatingFileHandler(
        'trading_bot.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if DISCORD_WEBHOOK_URL:
        discord_handler = DiscordHandler(DISCORD_WEBHOOK_URL)
        discord_handler.setFormatter(logging.Formatter("%(message)s"))
        discord_handler.setLevel(logging.INFO)
        logger.addHandler(discord_handler)
    return logger

logger = setup_logging()

# -------------------------------
# MODELS
# -------------------------------
class DualTradingDecision(BaseModel):
    decision: Literal["LONG", "SHORT", "HOLD"]
    percentage: int = Field(ge=0, le=100)
    reason: str

class Position(BaseModel):
    amount: float
    entry_price: float
    unrealized_profit: float

class Positions(BaseModel):
    long: Position
    short: Position

class TradeLog(BaseModel):
    timestamp: datetime
    long_btc_balance: float
    short_btc_balance: float
    usdt_balance: float
    long_entry_price: float
    short_entry_price: float
    long_decision: str
    short_decision: str
    long_percentage: float
    short_percentage: float
    long_reason: str
    short_reason: str
    reflection: str = ""
    portfolio_return: float
    btc_price_change: float
    total_assets: float
    portfolio_diff: float

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def adjust_quantity_to_step(qty: float, step: float = QUANTITY_STEP, min_qty: float = MIN_QUANTITY) -> float:
    if qty <= 0:
        return 0.0
    adjusted = math.floor(qty / step) * step
    return adjusted if adjusted >= min_qty else 0.0

def calculate_position_value(amount: float, price: float) -> float:
    return amount * price

def calculate_portfolio_return(current_value: float, previous_value: float) -> float:
    return ((current_value - previous_value) / previous_value * 100) if previous_value > 0 else 0.0

def read_strategy_file(filepath: str = "strategy.txt") -> Union[str, None]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading strategy file: {e}")
        return None

# -------------------------------
# INDICATORS
# -------------------------------
def add_indicators(df: pd.DataFrame, timeframe: str = '4h') -> pd.DataFrame:
    try:
        bb_settings = INDICATOR_SETTINGS['bollinger_bands']
        indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=bb_settings['window'], window_dev=bb_settings['window_dev'])
        df['bb_bbm'] = indicator_bb.bollinger_mavg()
        df['bb_bbh'] = indicator_bb.bollinger_hband()
        df['bb_bbl'] = indicator_bb.bollinger_lband()
        
        rsi_settings = INDICATOR_SETTINGS['rsi']
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=rsi_settings['window']).rsi()
        
        macd_settings = INDICATOR_SETTINGS['macd']
        macd = ta.trend.MACD(close=df['close'],
                             window_slow=macd_settings['window_slow'],
                             window_fast=macd_settings['window_fast'],
                             window_sign=macd_settings['window_sign'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        if timeframe in ['4h', '1d']:
            ma_settings = INDICATOR_SETTINGS['moving_averages']
            df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=ma_settings['sma_window']).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=ma_settings['ema_window']).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=ma_settings['ema_50_window']).ema_indicator()
            df['ema_200'] = ta.trend.EMAIndicator(close=df['close'], window=ma_settings['ema_200_window']).ema_indicator()

            atr_settings = INDICATOR_SETTINGS['atr']
            atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_settings['window'])
            df['atr'] = atr_indicator.average_true_range()
            df['atr_percent'] = (df['atr'] / df['close']) * 100

            stoch_rsi_settings = INDICATOR_SETTINGS['stoch_rsi']
            stoch_rsi = ta.momentum.StochRSIIndicator(close=df['close'],
                                                       window=stoch_rsi_settings['window'],
                                                       smooth1=stoch_rsi_settings['smooth_window'],
                                                       smooth2=stoch_rsi_settings['smooth_window'])
            df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

        if timeframe == '4h':
            adx_settings = INDICATOR_SETTINGS['adx']
            adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=adx_settings['window'])
            df['adx'] = adx.adx()
            df['di_plus'] = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()

            df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=INDICATOR_SETTINGS['cci']['window']).cci()
            df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=INDICATOR_SETTINGS['mfi']['window']).money_flow_index()
            df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=INDICATOR_SETTINGS['williams_r']['window']).williams_r()
        
        return df
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        return df

# -------------------------------
# SERVICES
# -------------------------------
class BinanceService:
    def __init__(self):
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            raise ValueError("Binance API keys not found")
        self.client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        
    def get_futures_account_balance(self):
        try:
            futures_account = self.client.futures_account_balance()
            usdt_balance = next((float(balance['balance']) for balance in futures_account if balance['asset'] == 'USDT'), 0)
            return usdt_balance
        except Exception as e:
            logger.error(f"Error getting futures balance: {e}")
            return 0

    def get_dual_side_positions(self, symbol=TRADING_SYMBOL):
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            dual_pos = {'long': {'amount': 0.0, 'entry_price': 0.0, 'unrealized_profit': 0.0},
                        'short': {'amount': 0.0, 'entry_price': 0.0, 'unrealized_profit': 0.0}}
            for p in positions:
                side_key = p["positionSide"].lower()
                if side_key in ['long', 'short']:
                    amt = abs(float(p['positionAmt']))
                    dual_pos[side_key] = {
                        'amount': amt,
                        'entry_price': float(p['entryPrice']),
                        'unrealized_profit': float(p['unRealizedProfit'])
                    }
            return dual_pos
        except Exception as e:
            logger.error(f"Error getting dual side positions: {e}")
            return dual_pos

    def get_multi_timeframe_data(self):
        try:
            market_data = {}
            for timeframe, config in TIMEFRAMES.items():
                interval = getattr(Client, config['interval'])
                market_data[timeframe] = self.get_futures_market_data(interval=interval, limit=config['limit'], timeframe=timeframe)
            return market_data
        except Exception as e:
            logger.error(f"Error getting multi-timeframe data: {e}")
            return None

    def get_futures_market_data(self, interval=Client.KLINE_INTERVAL_4HOUR, limit=300, timeframe='4h'):
        try:
            klines = self.client.futures_klines(symbol=TRADING_SYMBOL, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp','open','high','low','close','volume',
                'close_time','quote_av','trades','tb_base_av','tb_quote_av','ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return add_indicators(df, timeframe)
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
        
    def place_futures_order(self, side, quantity, position_side):
        try:
            self.client.futures_change_leverage(symbol=TRADING_SYMBOL, leverage=DEFAULT_LEVERAGE)
            time.sleep(1)
            mode_info = self.client.futures_get_position_mode()
            if not mode_info['dualSidePosition']:
                self.client.futures_change_position_mode(dualSidePosition=True)
                time.sleep(1)
            order = self.client.futures_create_order(
                symbol=TRADING_SYMBOL, type='MARKET', side=side, quantity=quantity, positionSide=position_side
            )
            if order is None:
                logger.warning("Order is None. Something went wrong.")
                return None
            logger.info(f"Order placed. OrderID={order['orderId']}")
            time.sleep(1)
            filled_order = self.client.futures_get_order(symbol=TRADING_SYMBOL, orderId=order['orderId'])
            logger.info(f"Order Executed - {side} {position_side}: 수량 {filled_order.get('executedQty')} BTC, 평균가격 {filled_order.get('avgPrice')} USDT")
            return filled_order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

class MarketDataService:
    def __init__(self):
        self.serpapi_key = SERPAPI_API_KEY

    def get_fear_and_greed_index(self):
        url = "https://api.alternative.me/fng/"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data['data'][0]
        except Exception as e:
            logger.error(f"Error fetching Fear and Greed Index: {e}")
            return None

class AIService:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.log_dir = Path('logs/gpt')
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate_reflection(self, trades_df, current_state):
        try:
            logger.info("Generating trading reflection...")
            if not trades_df:
                logger.warning("No trading history available")
                return None
            df = pd.DataFrame(trades_df).tail(10)
            if df.empty:
                logger.warning("Empty trading history")
                return None
            df_trades = df.to_json(orient='records', force_ascii=False)
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an AI trading assistant tasked with analyzing recent trading data and current market conditions to generate insights and improvements for future trading decisions."},
                    {"role": "user", "content": f"""
Recent trading data:
{df_trades}

Please analyze this data and provide:
1. A brief reflection on the recent trading decisions
2. Insights on what worked well and what didn't
3. Suggestions for improvement in future trading decisions
4. Any patterns or trends you notice in the market data

Limit your response to 250 words or less in korean
"""}
                ]
            )
            if not response or not response.choices:
                logger.error("Empty response from OpenAI during reflection generation")
                return None
            reflection = response.choices[0].message.content
            logger.info("Successfully generated trading reflection")
            self.log_gpt_interaction(timestamp=datetime.now(), current_state=current_state, strategy='', reflection=reflection)
            return reflection
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return None

    def get_trading_decision(self, current_state: Dict, strategy: str, reflection: str) -> Optional[DualTradingDecision]:
        try:
            if not self._validate_input_data(current_state, strategy, reflection):
                return None
            logger.info("Generating trading decision...")
            market_summary = self._format_market_data(current_state['market_data'])
            position_summary = self._format_position_data(current_state['positions'])
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": f"""
당신은 전문적인 비트코인 선물 트레이딩 전문가(포트폴리오 매니저)입니다. 
아래 내용을 바탕으로 현재 시점의 최적의 롱, 숏 포지션을 결정합니다.

가능한 포지션:
- LONG: 상승장 예상시 롱 포지션 진입 (기존 숏 포지션은 자동 청산)
- SHORT: 하락장 예상시 숏 포지션 진입 (기존 롱 포지션은 자동 청산)
- HOLD: 현재 포지션 유지 또는 관망

분석 시 필수적으로 고려해야 할 요소:
1) 기술적 지표 및 시장 데이터(1hour, 4hour, 1day)
2) Fear & Greed Index
3) 전반적인 시장 심리 (bullish/bearish, 변동성)
4) 최근 매매 성과 및 반성: {reflection}
5) 1x 레버리지 기준, 4 시간 마다 매매 결정
6) 트레이딩 전략 참고: {strategy}

추가 지시사항:
- 꼭 롱만 고집할 필요가 없습니다. 시장이 하락한다고 판단되면 적극적으로 숏 포지션도 고려하세요.
- percentage는 전체 자산 대비 포지션 크기입니다 (0-100).
- 리스크 관리를 고려하여 적절한 포지션 크기를 결정하세요.
- reason은 한국어 250자 이내로 설명해주세요
- HOLD에는 따로 퍼센트를 제시할 필요 없어요.
- 포지션 전환을 하지 않을때는 HOLD를 우선 고려
"""}, 
                    {"role": "user", "content": f"""
Current Status:
- Balance: {current_state['usdt_balance']:.2f} USDT
- Total Assets: {current_state['total_assets']:.2f} USDT
{position_summary}

Market Analysis:
{market_summary}
"""}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trading_decision",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "decision": {"type": "string", "enum": ["LONG", "SHORT", "HOLD"]},
                                "percentage": {"type": "integer"},
                                "reason": {"type": "string", "description": "트레이딩 결정의 이유"}
                            },
                            "required": ["decision", "percentage", "reason"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            if not response or not response.choices:
                logger.error("Empty response from OpenAI during decision generation")
                return None
            decision_content = response.choices[0].message.content
            logger.info(f"Generated trading decision: {decision_content}")
            trading_decision = DualTradingDecision.model_validate_json(decision_content)
            logger.info(f"Decision validated: {trading_decision.decision}, {trading_decision.percentage}%, Reason: {trading_decision.reason}")
            self.log_gpt_interaction(timestamp=datetime.now(), current_state=current_state, strategy=strategy, reflection=reflection, ai_response=trading_decision)
            return trading_decision
        except Exception as e:
            logger.error(f"Error getting trading decision: {e}")
            return None
        
    def _format_market_data(self, market_data: Dict) -> Dict:
        try:
            return {
                "price_summary": {
                    "current": market_data['price_summary']['current'],
                    "24h_high": market_data['price_summary']['24h_high'],
                    "24h_low": market_data['price_summary']['24h_low'],
                    "7d_price_change": market_data['price_summary']['7d_price_change']
                },
                "indicators": market_data['indicators'],
                "market_sentiment": market_data.get('market_sentiment', {}),
                "timeframes": market_data['timeframes']
            }
        except KeyError as e:
            logger.error(f"Missing market data field: {e}")
            return {}

    def _format_position_data(self, positions: Dict) -> Dict:
        return {
            "long": {
                "amount": positions['long']['amount'],
                "unrealized_profit": positions['long']['unrealized_profit'],
                "entry_price": positions['long']['entry_price']
            },
            "short": {
                "amount": positions['short']['amount'],
                "unrealized_profit": positions['short']['unrealized_profit'],
                "entry_price": positions['short']['entry_price']
            }
        }
    
    def _validate_input_data(self, current_state: Dict, strategy: str, reflection: str) -> bool:
        if not current_state or not isinstance(current_state, dict):
            logger.error("Invalid current_state")
            return False
        required_fields = ['usdt_balance', 'total_assets', 'positions', 'market_data']
        if not all(field in current_state for field in required_fields):
            logger.error(f"Missing required fields in current_state: {required_fields}")
            return False
        if not strategy:
            logger.error("Strategy is missing")
            return False
        return True

    def log_gpt_interaction(self, timestamp, current_state, strategy, reflection, ai_response=None):
        try:
            log_file = self.log_dir / timestamp.strftime('%Y%m%d') / f"gpt_interaction_{timestamp.strftime('%H%M%S')}.txt"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_content = f"""=== AI Trading Interaction Log ===
Timestamp: {timestamp}

[Reflection]
{reflection or "No reflection generated"}

[Trading Decision]
Decision: {ai_response.decision if ai_response else "N/A"}
Percentage: {ai_response.percentage if ai_response else "N/A"}
Reason: {ai_response.reason if ai_response else "N/A"}

[Current State]
Balance: {current_state['usdt_balance']:.2f} USDT
Total Assets: {current_state['total_assets']:.2f} USDT
Strategy: {strategy}

[Market Data]
{json.dumps(current_state['market_data'], indent=2, ensure_ascii=False, default=str)}"""
            log_file.write_text(log_content, encoding='utf-8')
            logger.info(f"GPT interaction logged to: {log_file}")
        except Exception as e:
            logger.error(f"Error logging GPT interaction: {e}")

# -------------------------------
# DatabaseService
# -------------------------------
class DatabaseService:
    def __init__(self, db_name=DATABASE_NAME):
        self.db_name = db_name
        self.log_dir = Path('logs/gpt')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self):
        try:
            with sqlite3.connect(self.db_name) as conn:
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        decision TEXT,
                        percentage REAL,
                        reason TEXT,
                        btc_balance REAL,      
                        usdt_balance REAL,     
                        entry_price REAL,      
                        reflection TEXT,        
                        portfolio_return REAL,  
                        btc_price_change REAL, 
                        total_assets REAL,     
                        portfolio_diff REAL    
                    )
                ''')
                c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades (timestamp DESC)")
                conn.commit()
                logger.info("데이터베이스가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 중 오류 발생: {e}")
            raise

    def get_recent_trades(self, limit: int = 10) -> list:
        try:
            query = "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?"
            with sqlite3.connect(self.db_name) as conn:
                c = conn.cursor()
                c.execute(query, (limit,))
                rows = c.fetchall()
                columns = [desc[0] for desc in c.description]
                trades_list = [dict(zip(columns, row)) for row in rows]
                logger.info(f"{len(trades_list)}개의 최근 거래 내역을 조회했습니다.")
                return trades_list
        except Exception as e:
            logger.error(f"최근 거래 내역 조회 중 오류 발생: {e}")
            return []

    def log_trade(self, timestamp: datetime, decision: str, percentage: float, reason: str,
                  btc_balance: float, usdt_balance: float, entry_price: float, reflection: str,
                  portfolio_return: float = 0.0, btc_price_change: float = 0.0, total_assets: float = 0.0,
                  portfolio_diff: float = 0.0):
        try:
            with sqlite3.connect(self.db_name) as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO trades (
                        timestamp, decision, percentage, reason,
                        btc_balance, usdt_balance, entry_price,
                        reflection, portfolio_return, btc_price_change,
                        total_assets, portfolio_diff
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat(), decision, percentage, reason,
                    btc_balance, usdt_balance, entry_price,
                    reflection, portfolio_return, btc_price_change,
                    total_assets, portfolio_diff
                ))
                conn.commit()
                logger.info(f"거래가 성공적으로 기록되었습니다 - 결정: {decision}, 비율: {percentage}%")
        except Exception as e:
            logger.error(f"거래 데이터 저장 중 오류 발생: {e}")
            raise

    def format_trade_history(self, trades_list: list) -> str:
        if not trades_list:
            return "거래 기록이 없습니다."
        formatted_trades = ""
        for trade in trades_list:
            formatted_trades += f"""
• {trade['timestamp']}
- 결정: {trade['decision']}
- 비율: {trade['percentage']}%
- BTC 잔고: {trade['btc_balance']:+.3f} BTC
- 진입가격: {trade['entry_price']:,.2f} USDT
- 총자산: {trade['total_assets']:,.2f} USDT
- 포트폴리오 수익률: {trade['portfolio_return']:+.2f}%\n"""
        return formatted_trades.strip()

# -------------------------------
# TRADING LOGIC
# -------------------------------
class TradingStrategy:
    def __init__(self, binance_service: BinanceService, database_service: DatabaseService,
                 market_data_service: MarketDataService, ai_service: AIService):
        self.binance = binance_service
        self.db = database_service
        self.market = market_data_service
        self.ai = ai_service

    def execute(self):
        try:
            account_status = self._get_account_status()
            if not account_status:
                return
            market_data = self._collect_market_data(account_status)
            if not market_data:
                return
            decision = self._generate_trading_decision(account_status, market_data)
            if not decision:
                return
            if not self._execute_decision(decision, account_status):
                return
            self._log_trade(decision, account_status, market_data["reflection"])
        except Exception as e:
            logger.error(f"Error executing trading strategy: {e}")
            raise

    def _get_account_status(self) -> Dict:
        try:
            market_data = self.binance.get_multi_timeframe_data()
            if not market_data or any(df.empty for df in market_data.values()):
                raise ValueError("Failed to get market data")
            current_price = float(market_data['4h']['close'].iloc[-1])
            usdt_balance = self.binance.get_futures_account_balance()
            positions = self.binance.get_dual_side_positions(TRADING_SYMBOL)
            long_value = positions['long']['amount'] * current_price
            short_value = positions['short']['amount'] * current_price
            total_position_value = long_value + short_value
            available_usdt = usdt_balance - total_position_value
            total_assets = total_position_value + available_usdt
            long_pnl = positions['long']['unrealized_profit']
            short_pnl = positions['short']['unrealized_profit']
            total_pnl = long_pnl + short_pnl
            total_pnl_percentage = (total_pnl / total_assets * 100) if total_assets > 0 else 0
            self._log_account_status(total_assets, total_pnl_percentage, positions,
                                      long_value, short_value, long_pnl, short_pnl,
                                      available_usdt, current_price)
            return {
                "usdt_balance": usdt_balance,
                "available_usdt": available_usdt,
                "total_assets": total_assets,
                "positions": positions,
                "current_price": current_price,
                "market_data": market_data
            }
        except Exception as e:
            logger.error(f"Error getting account status: {e}")
            return None

    def _collect_market_data(self, account_status: Dict) -> Dict:
        try:
            # 재호출 대신 _get_account_status()에서 가져온 market_data 재사용
            market_data = account_status["market_data"]
            if not market_data or any(df.empty for df in market_data.values()):
                logger.warning("Incomplete market data")
                return None
            optimized_data = self._prepare_market_data_for_ai(market_data['4h'], market_data['1h'], market_data['1d'])
            if not optimized_data:
                logger.error("Failed to prepare market data")
                return None
            fear_greed_index = self.market.get_fear_and_greed_index()
            if fear_greed_index:
                optimized_data["market_sentiment"] = {"fear_greed_index": fear_greed_index}
            recent_trades = self.db.get_recent_trades()
            reflection = self.ai.generate_reflection(recent_trades, account_status)
            return {"market_data": optimized_data, "reflection": reflection}
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return None

    def _generate_trading_decision(self, account_status: Dict, market_data: Dict) -> Optional[DualTradingDecision]:
        try:
            strategy = read_strategy_file()
            if not strategy:
                raise ValueError("Strategy file not found or empty")
            if not market_data or 'market_data' not in market_data:
                raise ValueError("Invalid market data structure")
            current_state = {
                "usdt_balance": account_status["available_usdt"],
                "total_assets": account_status["total_assets"],
                "positions": account_status["positions"],
                "market_data": market_data["market_data"]
            }
            self.ai.log_gpt_interaction(timestamp=datetime.now(), current_state=current_state,
                                        strategy=strategy, reflection=market_data.get("reflection"), ai_response=None)
            decision = self.ai.get_trading_decision(current_state, strategy, market_data.get("reflection", ""))
            if decision:
                logger.info(f"Trading Decision: {decision.decision}, {decision.percentage}%, Reason: {decision.reason}")
                return decision
            raise ValueError("Failed to get trading decision")
        except Exception as e:
            logger.error(f"Error generating trading decision: {e}")
            return None

    def _prepare_market_data_for_ai(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> Dict:
        try:
            latest_4h = df_4h.tail(24)
            latest_1h = df_1h.tail(24)
            latest_1d = df_1d.tail(7)
            def extract_key_data(df):
                return {
                    'open': df['open'].tolist(),
                    'close': df['close'].tolist(),
                    'high': df['high'].tolist(),
                    'low': df['low'].tolist(),
                    'volume': df['volume'].tolist(),
                }
            latest_indicators = {
                "bollinger": {
                    "middle": float(df_4h['bb_bbm'].iloc[-1]),
                    "upper": float(df_4h['bb_bbh'].iloc[-1]),
                    "lower": float(df_4h['bb_bbl'].iloc[-1])
                },
                "rsi": {
                    "4h": float(df_4h['rsi'].iloc[-1]),
                    "1d": float(df_1d['rsi'].iloc[-1])
                },
                "macd": {
                    "macd": float(df_4h['macd'].iloc[-1]),
                    "signal": float(df_4h['macd_signal'].iloc[-1]),
                    "diff": float(df_4h['macd_diff'].iloc[-1])
                },
                "moving_averages": {
                    "sma_20": float(df_4h['sma_20'].iloc[-1]),
                    "ema_12": float(df_4h['ema_12'].iloc[-1]),
                    "ema_50": float(df_4h['ema_50'].iloc[-1]),
                    "ema_200": float(df_4h['ema_200'].iloc[-1])
                },
                "trend_indicators": {
                    "atr": float(df_4h['atr'].iloc[-1]),
                    "adx": float(df_4h['adx'].iloc[-1]),
                    "cci": float(df_4h['cci'].iloc[-1]),
                    "mfi": float(df_4h['mfi'].iloc[-1])
                },
                "oscillators": {
                    "stoch_rsi_k": float(df_4h['stoch_rsi_k'].iloc[-1]),
                    "stoch_rsi_d": float(df_4h['stoch_rsi_d'].iloc[-1]),
                    "williams_r": float(df_4h['williams_r'].iloc[-1])
                }
            }
            market_data = {
                "timeframes": {
                    "4h": extract_key_data(latest_4h),
                    "1h": extract_key_data(latest_1h),
                    "1d": extract_key_data(latest_1d)
                },
                "indicators": latest_indicators,
                "price_summary": {
                    "current": float(df_4h['close'].iloc[-1]),
                    "24h_high": float(df_1h['high'].max()),
                    "24h_low": float(df_1h['low'].min()),
                    "24h_volume": float(df_1h['volume'].sum()),
                    "7d_price_change": float((df_1d['close'].iloc[-1] / df_1d['close'].iloc[0] - 1) * 100)
                }
            }
            return market_data
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return None

    def _log_account_status(self, total_assets, total_pnl_percentage, positions, long_value, short_value,
                            long_pnl, short_pnl, available_usdt, current_price):
        logger.info(
            f"Account Status\n"
            f"총 자산: {total_assets:.2f} US @ {total_pnl_percentage:+.2f}%\n"
            f"롱 포지션: {positions['long']['amount']} BTC (Value: {long_value:.2f} US, PnL: {(long_pnl/long_value*100 if long_value>0 else 0):+.2f}%)\n"
            f"숏 포지션: {positions['short']['amount']} BTC (Value: {short_value:.2f} US, PnL: {(short_pnl/short_value*100 if short_value>0 else 0):+.2f}%)\n"
            f"USDT 잔고: {available_usdt:.2f} US, 현재 BTC 가격: {current_price:.2f} US"
        )

    def _close_position(self, side: str, amount: float, current_price: float, unrealized_profit: float) -> bool:
        if amount > 0:
            logger.info(f"Closing {side.upper()} Position: Amount {amount} BTC, Value {amount * current_price:.2f} US, PnL: {unrealized_profit:+.2f} US")
            if side.lower() == 'short':
                return bool(self.binance.place_futures_order('BUY', amount, 'SHORT'))
            elif side.lower() == 'long':
                return bool(self.binance.place_futures_order('SELL', amount, 'LONG'))
        return True

    def _execute_decision(self, decision: DualTradingDecision, account_status: Dict) -> bool:
        try:
            current_price = account_status["current_price"]
            positions = account_status["positions"]
            usdt_balance = account_status["usdt_balance"]

            # 기존 포지션 청산
            if not self._close_position('short', positions['short']['amount'], current_price, positions['short']['unrealized_profit']):
                return False
            if not self._close_position('long', positions['long']['amount'], current_price, positions['long']['unrealized_profit']):
                return False

            # 새로운 포지션 진입
            qty = adjust_quantity_to_step((usdt_balance * decision.percentage / 100) / current_price)
            if qty > 0:
                if decision.decision == "LONG":
                    logger.info(f"Opening LONG Position: Qty {qty} BTC, Value {qty * current_price:.2f} US")
                    result = self.binance.place_futures_order('BUY', qty, 'LONG')
                else:
                    logger.info(f"Opening SHORT Position: Qty {qty} BTC, Value {qty * current_price:.2f} US")
                    result = self.binance.place_futures_order('SELL', qty, 'SHORT')
                return bool(result)
            elif decision.decision == "HOLD":
                logger.info(f"HOLD Position: 롱 {positions['long']['amount']} BTC, 숏 {positions['short']['amount']} BTC")
                return True
            return False
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            return False

    def _log_trade(self, decision: DualTradingDecision, account_status: Dict, reflection: str) -> None:
        try:
            timestamp = datetime.now()
            current_price = account_status["current_price"]
            positions = account_status["positions"]
            usdt_balance = account_status["usdt_balance"]
            available_usdt = account_status["available_usdt"]
            total_assets = account_status["total_assets"]
            self.db.log_trade(
                timestamp=timestamp,
                decision=decision.decision,
                percentage=decision.percentage,
                reason=decision.reason,
                btc_balance=positions['long']['amount'] - positions['short']['amount'],
                usdt_balance=available_usdt,
                entry_price=current_price,
                reflection=reflection,
                portfolio_return=0.0,
                btc_price_change=0.0,
                total_assets=total_assets,
                portfolio_diff=0.0
            )
            logger.info(f"Trade Complete: USDT 잔고 {usdt_balance:.2f}, 롱 {positions['long']['amount']} BTC, 숏 {positions['short']['amount']} BTC, 총자산 {total_assets:.2f} US")
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            raise

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bitcoin Trading Bot')
    parser.add_argument('--test', action='store_true', help='Run immediately for testing')
    args = parser.parse_args()
    
    trading_in_progress = False

    def trading_job():
        global trading_in_progress
        if trading_in_progress:
            logger.warning("Trading job is already in progress, skipping this run.")
            return
        try:
            trading_in_progress = True
            strategy = TradingStrategy(BinanceService(), DatabaseService(), MarketDataService(), AIService())
            strategy.execute()
        except Exception as e:
            logger.error(f"An error occurred in trading job: {e}")
        finally:
            trading_in_progress = False

    def setup_schedule():
        for time_str in SCHEDULE_TIMES:
            schedule.every().day.at(time_str).do(trading_job)
            logger.info(f"Scheduled trading job at {time_str}")

    def run_scheduled():
        logger.info("Starting Bitcoin Trading Bot in scheduled mode...")
        setup_schedule()
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise

    def run_immediate():
        logger.info("Starting Bitcoin Trading Bot in immediate mode...")
        try:
            trading_job()
            logger.info("Immediate execution completed")
        except Exception as e:
            logger.error(f"Error in immediate execution: {e}")
            raise

    if args.test:
        run_immediate()
    else:
        run_scheduled()
