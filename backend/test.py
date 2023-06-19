import requests 

BASE = "http://127.0.0.1:5000/"

response = requests.post(BASE + "prediction", {"query" : "write an algorithm"})
print(response.json())