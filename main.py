import os
import yfinance as yf
import pandas as pd
import ta
import asyncio
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, List, Optional, Set
import logging
from random import uniform
import json

# ===== Configuration =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOLS_CSV = os.getenv('SYMBOLS_CSV', 'under_100rs_stocks.csv')
DATA_DAYS = 90
CHECK_INTERVAL = 300  # 5 minutes (set to 60 for testing)
RISK_REWARD_RATIO = 2
MAX_RETRIES = 2
TEST_MODE = False  # Set to True for testing during off-hours
STALE_DATA_THRESHOLD = 15  # minutes
MAX_FAILED_SYMBOLS = 50
BATCH_SIZE = 20

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('yfinance').setLevel(logging.WARNING)  # Reduce yfinance noise

class StockTradingBot:
    def __init__(self):
        self.app = None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.signal_cache = {}
        self.performance_stats = {}
        self.market_timezone = ZoneInfo("Asia/Kolkata")
        self.symbols = []
        self.failed_symbols = set()
        self.symbol_retries = {}
        self.paused = False

    # ===== Core Methods =====
    async def initialize(self):
        """Initialize the bot with all components"""
        try:
            self.symbols = await self.load_symbols()
            if not self.symbols:
                logger.error("No symbols loaded - check your CSV file")
                return False
            
            self.app = Application.builder().token(TELEGRAM_TOKEN).build()
            self.app.add_handler(CommandHandler("pause", self.pause))
            self.app.add_handler(CommandHandler("resume", self.resume))
            
            await self.send_startup_message()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return False

    async def load_symbols(self) -> List[str]:
        """Load stock symbols from CSV with duplicate .NS check"""
        try:
            if not os.path.exists(SYMBOLS_CSV):
                logger.error(f"Symbols file not found: {SYMBOLS_CSV}")
                return []

            df = pd.read_csv(SYMBOLS_CSV)
            if 'Symbol' not in df.columns:
                logger.error("CSV missing 'Symbol' column")
                return []

            symbols = df['Symbol'].dropna().unique().tolist()
            clean_symbols = []
            
            for s in symbols:
                if not isinstance(s, str):
                    continue
                    
                s = s.strip().upper()
                # Remove existing .NS if present
                if s.endswith('.NS'):
                    s = s[:-3]
                clean_symbols.append(f"{s}.NS")

            logger.info(f"Loaded {len(clean_symbols)} symbols")
            return clean_symbols
        except Exception as e:
            logger.error(f"Failed to load symbols: {str(e)}")
            return []

    async def run(self):
        """Main trading loop"""
        if not await self.initialize():
            return

        logger.info(f"==== {'TEST MODE' if TEST_MODE else 'LIVE MODE'} ====")
        
        try:
            while True:
                try:
                    if self.paused:
                        await asyncio.sleep(60)
                        continue

                    if not self.is_market_open() and not TEST_MODE:
                        logger.debug("Market closed - sleeping")
                        await asyncio.sleep(CHECK_INTERVAL)
                        continue

                    active_symbols = [s for s in self.symbols if s not in self.failed_symbols]
                    logger.info(f"Scanning {len(active_symbols)} active symbols")
                    
                    # Process in batches
                    for i in range(0, len(active_symbols), BATCH_SIZE):
                        batch = active_symbols[i:i+BATCH_SIZE]
                        batch_data = await self.fetch_batch_data(batch)
                        
                        for symbol, data in batch_data.items():
                            if data is None:
                                continue
                                
                            signal, price, notes = self.generate_signal(symbol, data)
                            if signal in ('BUY', 'SELL'):
                                await self.send_alert(signal, symbol, price, notes)

                    await asyncio.sleep(CHECK_INTERVAL)

                except Exception as e:
                    logger.error(f"Scanning error: {str(e)}")
                    await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    # ===== Telegram Commands =====
    async def pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command"""
        self.paused = True
        await update.message.reply_text("⏸️ Bot paused")

    async def resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        self.paused = False
        await update.message.reply_text("▶️ Bot resumed")

    # ===== Market Methods =====
    async def get_market_trend(self) -> str:
        """Determine overall market trend"""
        try:
            data = yf.download('^NSEI', period='2d', progress=False, auto_adjust=True)
            if len(data) < 2:
                return "Neutral (No Data)"
                
            last_close = data['Close'].iloc[-1].item()
            prev_close = data['Close'].iloc[-2].item()
            
            if last_close > prev_close:
                return "Bullish"
            elif last_close < prev_close:
                return "Bearish"
            return "Neutral"
        except Exception as e:
            logger.error(f"Market trend error: {str(e)}")
            return "Neutral (Error)"

    def is_market_open(self) -> bool:
        """Check if market is open"""
        if TEST_MODE:
            return True
        now = datetime.now(self.market_timezone)
        return (now.weekday() < 5 and 
                dtime(9, 15) <= now.time() <= dtime(15, 30))

    # ===== Data Methods =====
    async def fetch_batch_data(self, symbols: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch data for multiple symbols"""
        results = {}
        async with httpx.AsyncClient() as client:
            tasks = {symbol: self.fetch_stock_data(symbol, client) for symbol in symbols}
            completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for symbol, result in zip(tasks.keys(), completed):
                if isinstance(result, Exception):
                    logger.debug(f"Batch failed for {symbol}: {str(result)}")
                    results[symbol] = None
                else:
                    results[symbol] = result
        return results

    async def fetch_stock_data(self, symbol: str, client: httpx.AsyncClient) -> Optional[pd.DataFrame]:
        """Fetch data for single symbol with improved error handling"""
        try:
            # Clean symbol (remove duplicate .NS if any)
            clean_symbol = symbol.replace('.NS.NS', '.NS')
            
            # Use daily data when market is closed
            period = "1d" if self.is_market_open() else f"{DATA_DAYS}d"
            interval = "15m" if self.is_market_open() else "1d"
            
            data = await asyncio.to_thread(
                yf.download,
                clean_symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                logger.debug(f"No data for {clean_symbol} - possibly delisted")
                self.failed_symbols.add(symbol)
                return None
                
            return data
        except Exception as e:
            logger.debug(f"Data fetch failed for {symbol}: {str(e)}")
            self.failed_symbols.add(symbol)
            return None

    # ===== Signal Generation =====
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """Generate trading signal with robust data handling"""
        try:
            # Validate data
            if data.empty or len(data) < 5 or 'Close' not in data.columns or 'Volume' not in data.columns:
                return "HOLD", 0, ["Insufficient data"]

            # Convert to 1D Series explicitly
            close = pd.Series(data['Close'].values.ravel())
            volume = pd.Series(data['Volume'].values.ravel())
            
            # Calculate indicators with safety checks
            try:
                rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
                macd = ta.trend.MACD(close=close)
                sma20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
                
                current = {
                    'price': float(close.iloc[-1]),
                    'rsi': float(rsi.iloc[-1]),
                    'macd_line': float(macd.macd().iloc[-1]),
                    'signal_line': float(macd.macd_signal().iloc[-1]),
                    'sma20': float(sma20.iloc[-1]),
                    'volume': float(volume.iloc[-1]),
                    'avg_volume': float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
                }
            except Exception as e:
                logger.warning(f"Indicator calculation failed for {symbol}: {str(e)}")
                return "HOLD", 0, ["Indicator error"]

            # Signal logic
            notes = []
            buy_score = sell_score = 0

            # Buy conditions
            if current['rsi'] < 30:
                buy_score += 2
                notes.append(f"RSI {current['rsi']:.1f} (Oversold)")
            if current['macd_line'] > current['signal_line']:
                buy_score += 2
                notes.append("MACD Bullish")
            if current['price'] > current['sma20']:
                buy_score += 1
                notes.append("Price > SMA20")
            if current['volume'] > 1.5 * current['avg_volume']:
                buy_score += 1
                notes.append("Volume spike")

            # Sell conditions
            if current['rsi'] > 70:
                sell_score += 2
                notes.append(f"RSI {current['rsi']:.1f} (Overbought)")
            if current['macd_line'] < current['signal_line']:
                sell_score += 2
                notes.append("MACD Bearish")
            if current['price'] < current['sma20']:
                sell_score += 1
                notes.append("Price < SMA20")

            # Generate signal
            if buy_score >= 5 and buy_score > sell_score:
                return "BUY", current['price'], notes
            elif sell_score >= 5 and sell_score > buy_score:
                return "SELL", current['price'], notes
            
            return "HOLD", current['price'], ["No strong signal"]
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {str(e)}")
            return "ERROR", 0, [f"System error: {str(e)}"]

    # ===== Alert Methods =====
    async def send_startup_message(self):
        """Send initial status message"""
        try:
            market_status = "OPEN" if self.is_market_open() else "CLOSED"
            market_trend = await self.get_market_trend()
            
            message = [
                "🚀 <b>Stock Signal Bot Activated</b>",
                f"• Market Status: {market_status}",
                f"• Market Trend: {market_trend}",
                f"• Tracking {len(self.symbols)} symbols",
                f"• Failed Symbols: {len(self.failed_symbols)}",
                f"• Next scan in {CHECK_INTERVAL//60} minutes",
                f"• Mode: {'TEST' if TEST_MODE else 'LIVE'}"
            ]

            await self.app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="\n".join(message),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Startup message failed: {str(e)}")

    async def send_alert(self, signal: str, symbol: str, price: float, notes: List[str]):
        """Send trading alert"""
        try:
            emoji = "🟢" if signal == "BUY" else "🔴"
            message = [
                f"{emoji} <b>{signal} {symbol}</b> {emoji}",
                f"Price: ₹{price:.2f}",
                "",
                "<b>Rationale:</b>",
                *notes,
                "",
                f"<i>{datetime.now(self.market_timezone).strftime('%Y-%m-%d %H:%M:%S')}</i>"
            ]

            await self.app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="\n".join(message),
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Alert failed for {symbol}: {str(e)}")

    async def shutdown(self):
        """Cleanup resources"""
        try:
            await self.client.aclose()
            if self.app:
                await self.app.shutdown()
            logger.info("Bot shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")

async def main():
    bot = StockTradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Missing Telegram credentials in .env file")
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal startup error: {str(e)}")
