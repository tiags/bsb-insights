import pandas as pd
from stockdex import Ticker
import yfinance as yf
from datetime import datetime
import numpy as np
import time
import random
import os
import shutil
from datetime import timedelta
import subprocess
import sys
import sqlite3
import nltk
import requests
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
from logging.handlers import RotatingFileHandler
import os

# Setup log directory and file
log_dir = os.path.join(os.getcwd(), "Misc", "Logging")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "scrapes.log")

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# File handler with rotation
file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(formatter)

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

finnhub_api = "***"

start = time.time()
conn = sqlite3.connect("bsb.db")

def is_mac_awake():
    try:
        pmset_output = subprocess.check_output(["pmset", "-g"]).decode("utf-8")
        if " Sleep" in pmset_output and "0" not in pmset_output:
            return False
        return True
    except Exception as e:
        logger.exception("Failed")
        return False

# Exit if Mac is asleep
if not is_mac_awake():
    ...

# Exit if it's the weekend
if datetime.today().weekday() >= 5:
    ...

def notify(title, message):
    os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')

def log_start():
    start_time = datetime.now().strftime("%I:%M %p")
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    notify("BSB Task", f"Started at {start_time}")
    logger.info(f"▶️ BSB Scrape Started at {timestamp} {start_time}")

log_start()

# Load Excel sheet
excel_sheet = "...INPUTS.xlsx"
df = pd.read_excel(excel_sheet, sheet_name="Everything", header=0)
pd.options.display.float_format = '{:.2f}'.format

# Parameters
BATCH_SIZE = 50  # Process 50 stocks per batch
RETRIES = 3  # Number of retry attempts if a request fails
WAIT_TIME = 1  # Base wait time (will increase exponentially)
RESULTS_FILE = "Results.csv"

# Initialize DataFrame to store results
results = pd.DataFrame(columns=[
    ...
])

first_write = not os.path.exists(RESULTS_FILE)

def exponential_backoff(attempt):
    max_wait = 60
    wait_time = min(WAIT_TIME * (2 ** attempt) + random.uniform(0, 2), max_wait)
    logger.info(f"🕒 Waiting {wait_time:.2f} seconds before retrying...")
    time.sleep(wait_time)

def convert_to_numeric(value):
    try:
        if pd.notna(value): 
            if 'B' in str(value) or 'b' in str(value): 
                return float(str(value).replace('B', '').replace('b', '').strip()) * 1e9
            elif 'M' in str(value) or 'm' in str(value): 
                return float(str(value).replace('M', '').replace('m', '').strip()) * 1e6
            elif 'T' in str(value) or 't' in str(value): 
                return float(str(value).replace('T', '').replace('t', '').strip()) * 1e12
            elif 'K' in str(value) or 'k' in str(value): 
                return float(str(value).replace('K', '').replace('k', '').strip()) * 1e3
            else: 
                return float(value)
        else:
            return None
    except Exception as e:
        logger.error(f"Error converting value {value}: {e}")
        
        return None

def get_stock_info(symbol):
    for attempt in range(RETRIES):
        try:
            ...

def get_avg_sentiment(symbol, analyzer):
    try:
        ...
    except Exception as e:
        logger.error(f"[ERROR] {symbol} - RSS sentiment fetch failed: {e}")
        return None
    
def get_insider_activity(symbol, finnhub_api):
    try:
        ...
        return buys, sells
    except Exception as e:
        logger.error(f"[ERROR] {symbol} - Insider fetch failed: {e}")
        return 0, 0

def get_days_until_earnings(symbol, finnhub_api):
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        future = (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')
        ...
        return None
    
def debug_missing(symbol, var_name, value):
    if isinstance(value, pd.DataFrame) and value.empty:
        logger.error(f"[ERROR] {symbol} - Unable to retrieve {var_name} (Empty DataFrame)")
    elif isinstance(value, pd.Series) and value.empty:
        logger.error(f"[ERROR] {symbol} - Unable to retrieve {var_name} (Empty Series)")
    elif value is None or (not isinstance(value, (pd.DataFrame, pd.Series)) and pd.isna(value)):
        logger.error(f"[ERROR] {symbol} - Unable to retrieve {var_name} (Missing Value)")

def fetch_data(symbol):
        ...
        return tuple(return_values)
    
def calculate_dcf(symbol, cash_flow, market_cap, total_debt, total_cash, shares_outstanding):
    discount_rate = 0.1
    growth_rate = 0.05
    years_to_project = 5
    terminal_growth_rate = 0.02
    
    ...
    
    return intrinsic_value, presentvalue_fcf_5

def calculate_mss_mvs(market_breadth):
    ...
    
def update_market_trend_csv(market_breadth):
    ...

def compute_earnings_growth(net_income_list, years):
    ...

def calculate_rsi(data, period=14):
    ...

def calculate_macd(data, short=12, long=26, signal=9):
    ...

def calculate_bollinger_bands(data, window=20):
    ...
    
def calculate_moving_averages(data, short_window=50, long_window=200):
    ...

def check_ma_crossovers(sma_50, sma_200):
    ...

def get_options_data(symbol):
    ...
    
def transform_and_clean_data(df):
    ...

def process_symbols(df):
    ...

def save_results_to_csv(df):
    ...
    
def save_results_to_sql(df):
    ...
        
def alert_and_notify(df):
    ...

def backup_results_file():
    ...

def log_complete():
    ...

backup_results_file()
df = process_symbols(df)
df = df.sort_values("Timestamp").drop_duplicates("Symbol", keep="last")  # 🧹 Deduplicate
save_results_to_csv(df)
save_results_to_sql(df)
alert_and_notify(df)

now = datetime.now().strftime("%I:%M %p")
notify("BSB Task", f"Backup/Scrape completed at {now}")
log_complete()
logger.info("All Data In Million!")
logger.info(f"📊 Current total in bsb_data: {pd.read_sql('SELECT COUNT(*) FROM bsb_data', conn).iloc[0, 0]}")
conn.close()
logger.info("SQLite connection closed")
end = time.time()
logger.info(f"Total time: {end - start:.2f} seconds")