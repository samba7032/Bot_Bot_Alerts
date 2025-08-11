#!/usr/bin/env python3
"""
main.py - StockSignalBot
Rewritten to fix pandas ambiguous-series and future float warnings,
optimize batch downloads via yfinance, and include ATR-based stop/target.
"""
import os
import asyncio
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import httpx
import yfinance as yf
import pandas as pd
import ta
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ===== Configuration =====
load_dotenv()
TELEGRAM_TOKEN = '8305784916:AAE2UP_4CxpYVHxfpD1yFBk8hi3uU-vd32I'
TELEGRAM_CHAT_ID = '1020815701'
SYMBOLS_CSV = os.getenv("SYMBOLS_CSV", "under_100rs_stocks.csv")

DATA_DAYS = int(os.getenv("DATA_DAYS", "90"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "600"))  # seconds between scans
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
TEST_MODE = os.getenv("TEST_MODE", "False").lower() in ("1", "true", "yes")

RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "2.0"))
STOP_ATR_MULTIPLIER = float(os.getenv("STOP_ATR_MULTIPLIER", "1.5"))

MARKET_TZ = ZoneInfo("Asia/Kolkata")

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("StockSignalBot")
logging.getLogger("yfinance").setLevel(logging.WARNING)

# ===== Bot Class =====
class StockTradingBot:
    def __init__(self):
        self.app = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.symbols: List[str] = []
        self.failed_symbols = set()
        self.paused = False
        self.market_timezone = MARKET_TZ

    # ---- initialization and symbol loading ----
    async def initialize(self) -> bool:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in environment")
            return False

        self.symbols = await self.load_symbols()
        if not self.symbols:
            logger.error("No symbols loaded.")
            return False

        # Telegram app
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.app.add_handler(CommandHandler("pause", self.pause))
        self.app.add_handler(CommandHandler("resume", self.resume))
        self.app.add_handler(CommandHandler("status", self.status))

        await self.send_startup_message()
        return True

    async def load_symbols(self) -> List[str]:
        if not os.path.exists(SYMBOLS_CSV):
            logger.error("Symbols CSV not found: %s", SYMBOLS_CSV)
            return []
        try:
            df = pd.read_csv(SYMBOLS_CSV)
            if "Symbol" not in df.columns:
                logger.error("CSV missing 'Symbol' column")
                return []
            symbols = []
            for s in df["Symbol"].dropna().unique():
                if not isinstance(s, str):
                    continue
                s = s.strip().upper()
                if s.endswith(".NS"):
                    s = s[:-3]
                symbols.append(f"{s}.NS")
            logger.info("Loaded %d symbols", len(symbols))
            return symbols
        except Exception as e:
            logger.error("Failed to load symbols CSV: %s", e)
            return []

    # ---- main loop ----
    async def run(self):
        if not await self.initialize():
            return

        logger.info("==== %s MODE ====", "TEST" if TEST_MODE else "LIVE")

        try:
            while True:
                if self.paused:
                    logger.info("Paused. Sleeping 60s...")
                    await asyncio.sleep(60)
                    continue

                if not self.is_market_open() and not TEST_MODE:
                    logger.debug("Market closed. Sleeping for CHECK_INTERVAL.")
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                active_symbols = [s for s in self.symbols if s not in self.failed_symbols]
                logger.info("Scanning %d active symbols", len(active_symbols))

                # batch download via yfinance to reduce per-symbol latency
                for i in range(0, len(active_symbols), BATCH_SIZE):
                    batch = active_symbols[i : i + BATCH_SIZE]
                    batch_data = await asyncio.to_thread(self.batch_download, batch)
                    for sym in batch:
                        df = batch_data.get(sym)
                        if df is None:
                            continue
                        signal, price, notes = self.generate_signal(sym, df)
                        if signal in ("BUY", "SELL"):
                            await self.send_alert(signal, sym, price, notes)

                await asyncio.sleep(CHECK_INTERVAL)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    # ---- market helpers ----
    def is_market_open(self) -> bool:
        if TEST_MODE:
            return True
        now = datetime.now(self.market_timezone)
        return now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 30)

    async def get_market_trend(self) -> str:
        try:
            data = await asyncio.to_thread(yf.download, "^NSEI", period="2d", progress=False, auto_adjust=True)
            if data is None or data.empty or "Close" not in data.columns or len(data) < 2:
                return "Neutral (No Data)"
            # pick last close scalar
            last = data["Close"].iloc[-1].item()
            prev = data["Close"].iloc[-2].item()
            if last > prev:
                return "Bullish"
            if last < prev:
                return "Bearish"
            return "Neutral"
        except Exception as e:
            logger.error("Market trend error: %s", e)
            return "Neutral (Error)"

    # ---- data fetching ----
    def batch_download(self, symbols: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Use yfinance.download once per batch to get all tickers.
        Returns dict: symbol -> DataFrame (or None)
        """
        results: Dict[str, Optional[pd.DataFrame]] = {s: None for s in symbols}
        if not symbols:
            return results

        tickers_str = " ".join(symbols)
        try:
            # group_by='ticker' gives nested columns when multiple tickers requested
            raw = yf.download(tickers=tickers_str, period=f"{DATA_DAYS}d", interval="1d", group_by="ticker", auto_adjust=True, progress=False)
            # If only one ticker requested, yfinance returns a DataFrame w/o ticker-level columns
            if len(symbols) == 1:
                s = symbols[0]
                if raw is None or raw.empty:
                    results[s] = None
                else:
                    results[s] = raw.copy()
                return results

            # Multiple tickers - raw.columns is MultiIndex (ticker, field)
            for s in symbols:
                try:
                    # yfinance may use the ticker string exactly as provided in columns
                    if (s in raw.columns.get_level_values(0)):
                        df = raw[s].copy()
                        if df is None or df.empty:
                            results[s] = None
                        else:
                            results[s] = df
                    else:
                        # Sometimes ticker keys are slightly different (e.g. with suffixes)
                        # Fallback: check for a single-level df (unlikely here) or mark as failed
                        results[s] = None
                except Exception:
                    results[s] = None
        except Exception as e:
            logger.debug("Batch download exception: %s", e)
            # On failure, try per-symbol fallback (slower)
            for s in symbols:
                try:
                    df = yf.download(s, period=f"{DATA_DAYS}d", interval="1d", auto_adjust=True, progress=False)
                    results[s] = df if not df.empty else None
                except Exception:
                    results[s] = None
        # Mark missing as failed_symbols later in calling code
        for s, df in results.items():
            if df is None:
                logger.debug("No data for %s in this batch", s)
                self.failed_symbols.add(s)
        return results

    # ---- signal generation ----
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        Returns (signal, price, notes)
        signal: "BUY","SELL","HOLD","ERROR"
        """
        try:
            if df is None or df.empty or "Close" not in df.columns or len(df) < 5:
                return "HOLD", 0.0, ["Insufficient data"]

            # Ensure required columns exist
            for col in ("High", "Low", "Open", "Volume"):
                if col not in df.columns:
                    df[col] = df["Close"]

            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            volume = df["Volume"].astype(float)

            # Safely extract scalars using .iloc[-1].item()
            price = close.iloc[-1].item()
            prev_close = close.iloc[-2].item() if len(close) > 1 else price

            # Compute indicators with try/fallbacks
            try:
                rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
                rsi = rsi_series.iloc[-1].item() if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50.0
            except Exception:
                rsi = 50.0

            try:
                macd = ta.trend.MACD(close=close)
                macd_line = macd.macd().iloc[-1].item() if not macd.macd().empty else 0.0
                macd_signal = macd.macd_signal().iloc[-1].item() if not macd.macd_signal().empty else 0.0
                macd_hist = macd.macd_diff().iloc[-1].item() if not macd.macd_diff().empty else 0.0
            except Exception:
                macd_line = macd_signal = macd_hist = 0.0

            def safe_sma(window: int) -> float:
                try:
                    s = ta.trend.SMAIndicator(close=close, window=window).sma_indicator()
                    return s.iloc[-1].item() if not s.empty and not pd.isna(s.iloc[-1]) else 0.0
                except Exception:
                    return 0.0

            sma20 = safe_sma(20)
            sma50 = safe_sma(50)
            sma200 = safe_sma(200)

            try:
                atr_series = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
                atr = atr_series.iloc[-1].item() if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else 0.0
            except Exception:
                atr = 0.0

            notes: List[str] = []

            # Quality checks
            if price <= 0 or pd.isna(price):
                return "HOLD", 0.0, ["Invalid price"]

            # Strategy logic (example conservative combo)
            buy_condition = (rsi < 35) and (macd_line > macd_signal) and (price > sma20)
            sell_condition = (rsi > 65) and (macd_line < macd_signal) and (price < sma20)

            # ATR-based stop distance
            stop_distance = (atr * STOP_ATR_MULTIPLIER) if (atr and atr > 0) else max(0.015 * price, 0.01 * price)

            if buy_condition:
                stop = max(0.0, price - stop_distance)
                risk_amount = price - stop
                target = price + (risk_amount * RISK_REWARD_RATIO)
                notes.append(f"BUY conditions met: RSI {rsi:.1f}, MACD bullish (line>{'signal' if macd_line>macd_signal else 'no'})")
                notes.append(f"SMA20 {sma20:.2f} | SMA50 {sma50:.2f} | SMA200 {sma200:.2f}")
                notes.append(f"ATR {atr:.4f} | stop_distance {stop_distance:.4f}")
                notes.append(f"Stop: ₹{stop:.2f} | Target: ₹{target:.2f} | Risk: ₹{risk_amount:.2f} | R:R {RISK_REWARD_RATIO:.2f}:1")
                return "BUY", price, notes

            if sell_condition:
                stop = price + stop_distance
                risk_amount = stop - price
                target = price - (risk_amount * RISK_REWARD_RATIO)
                notes.append(f"SELL conditions met: RSI {rsi:.1f}, MACD bearish (line<{ 'signal' if macd_line<macd_signal else 'no'})")
                notes.append(f"SMA20 {sma20:.2f} | SMA50 {sma50:.2f} | SMA200 {sma200:.2f}")
                notes.append(f"ATR {atr:.4f} | stop_distance {stop_distance:.4f}")
                notes.append(f"Stop: ₹{stop:.2f} | Target: ₹{target:.2f} | Risk: ₹{risk_amount:.2f} | R:R {RISK_REWARD_RATIO:.2f}:1")
                return "SELL", price, notes

            # Default - no trade
            notes.append(f"No strong signal - RSI {rsi:.1f}, MACD hist {macd_hist:.4f}")
            return "HOLD", price, notes

        except Exception as e:
            logger.exception("Signal generation error for %s: %s", symbol, e)
            return "ERROR", 0.0, [f"Signal error: {e}"]

    # ---- alerts ----
    async def send_startup_message(self):
        try:
            trend = await self.get_market_trend()
            market_status = "OPEN" if self.is_market_open() else "CLOSED"
            message = [
                "🚀 <b>Stock Signal Bot Activated</b>",
                f"• Market Status: {market_status}",
                f"• Market Trend: {trend}",
                f"• Tracking: {len(self.symbols)} symbols",
                f"• Failed symbols: {len(self.failed_symbols)}",
                f"• Next scan: {CHECK_INTERVAL // 60} minutes",
                f"• Mode: {'TEST' if TEST_MODE else 'LIVE'}",
                f"• ATR stop multiplier: {STOP_ATR_MULTIPLIER}  |  R:R: {RISK_REWARD_RATIO}"
            ]
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="\n".join(message), parse_mode="HTML")
        except Exception as e:
            logger.error("Failed to send startup message: %s", e)

    async def send_alert(self, signal: str, symbol: str, price: float, notes: List[str]):
        try:
            emoji = "🟢" if signal == "BUY" else "🔴"
            header = f"{emoji} <b>{signal} {symbol}</b> {emoji}"
            body = [
                header,
                f"Price: ₹{price:.2f}",
                "",
                "<b>Rationale & Trade Plan</b>:",
                *notes,
                "",
                f"<i>{datetime.now(self.market_timezone).strftime('%Y-%m-%d %H:%M:%S')}</i>"
            ]
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="\n".join(body), parse_mode="HTML")
            logger.info("Sent %s alert for %s at ₹%.2f", signal, symbol, price)
        except Exception as e:
            logger.error("Failed to send alert for %s: %s", symbol, e)

    # ---- telegram commands ----
    async def pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = True
        await update.message.reply_text("⏸️ Bot paused")

    async def resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = False
        await update.message.reply_text("▶️ Bot resumed")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        market_status = "OPEN" if self.is_market_open() else "CLOSED"
        await update.message.reply_text(
            f"Status:\n• Market: {market_status}\n• Symbols: {len(self.symbols)}\n• Failed: {len(self.failed_symbols)}"
        )

    # ---- shutdown ----
    async def shutdown(self):
        try:
            await self.http_client.aclose()
            if self.app:
                await self.app.shutdown()
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error("Shutdown error: %s", e)


# ===== entrypoint =====
async def main():
    bot = StockTradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("Fatal error: %s", e)

