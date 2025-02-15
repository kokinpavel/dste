import requests

url = "http://127.0.0.1:8800/predict/"
data = {
    "features": {
        "age": 49,
        "job": "blue-collar",
        "marital": "married",
        "education": "basic.9y",
        "default": "unknown",
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "month": "nov",
        "day_of_week": "wed",
        "duration": 5,
        "campaign": 4,
        "previous": 0,
        "poutcome": "new",
    }
}

response = requests.post(url, json=data)
print(response.json())  # Output: {'prediction': 1}
