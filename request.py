import requests


url = "http://127.0.0.1:8000/predict"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
data = {
    "PULocationID": "1",
    "DOLocationID": "2",
    "trip_distance": 2
}
response = requests.post(url, headers=headers, json=data)
print(response.json())