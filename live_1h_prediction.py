import time
from datetime import datetime
import requests
import yfinance as yf

URL = "https://mnq-trading-api-699539648647.us-central1.run.app/predict"

SLEEP_INTERVAL = 3600 

print("Starting automated MNQ 1-hour prediction loop...")
print(f"Target API: {URL}")
print("Press Ctrl + C in your terminal to stop the bot at any time.\n")

while True:
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching latest 1-hour MNQ data...")
        
        ticker = yf.Ticker("MNQ=F")
        df = ticker.history(period="5d", interval="1h")
        
        if df.empty:
            print("Warning: Could not fetch market data. Retrying next cycle.")
        else:
            latest_bar = df.iloc[-1]
            
            payload = {
                "features": {
                    "open": float(latest_bar["Open"]),
                    "high": float(latest_bar["High"]),
                    "low": float(latest_bar["Low"]),
                    "close": float(latest_bar["Close"]),
                    "volume": int(latest_bar["Volume"])
                }
            }

            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                print(" Prediction Successful:")
                print(response.json())
            else:
                print(f" API Error (Status {response.status_code}): {response.text}")
                
    except Exception as e:
        print(f" Error occurred during execution: {e}")

    print(f"Sleeping for {SLEEP_INTERVAL // 60} minutes until the next check...\n")
    time.sleep(SLEEP_INTERVAL)