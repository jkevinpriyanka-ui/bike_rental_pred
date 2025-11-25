import requests

data = {
    "season": 3,
    "holiday": 0,
    "weekday": 2,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.5,
    "hum": 0.6,
    "windspeed": 0.1,
    "year": 2012,
    "month": 8
}

url = "http://127.0.0.1:5000/predict"
response = requests.post(url, json=data)
print(response.json())
