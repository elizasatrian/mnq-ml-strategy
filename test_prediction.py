import requests

# Your live Cloud Run service URL
url = "https://mnq-trading-api-699539648647.us-central1.run.app/predict"

# Sample feature dictionary (make sure the keys match your trained model's feature columns)
payload = {
    "features": {
        "open": 18250.5,
        "high": 18275.0,
        "low": 18240.0,
        "close": 18260.0,
        "volume": 12500
        # Add any other specific technical features your no_hour model expects
    }
}

print("Sending request to live Cloud Run API...")
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())