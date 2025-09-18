#!/usr/bin/env python3
"""
Quick test script to verify your FastAPI chatbot endpoint is working.
Run this from your project root directory.
"""

import requests
import json

def test_chatbot():
    url = "http://127.0.0.1:8000/chatbot/query"
    payload = {"question": "water"}
    headers = {"Content-Type": "application/json"}
    
    print(f"🧪 Testing chatbot endpoint...")
    print(f"URL: {url}")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Answer: {data.get('answer')}")
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
        print("Make sure FastAPI is running with: uvicorn main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    test_chatbot()