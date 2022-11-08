import requests

url = "http://localhost:9696/predict"

data = {'iucr': '1477',
 'primary_type': 'WEAPONS VIOLATION',
 'description': 'RECKLESS FIREARM DISCHARGE',
 'location_description': 'PARK PROPERTY',
 'fbi_code': '15',
 'zip': '066XX',
 'street': 'N, WESTERN, AVE',
 'domestic': 0,
 'beat': 2412,
 'district': 24,
 'ward': 50,
 'community_area': 2,
 'latitude': 42.001822361,
 'longitude': -87.689987495,
 'hour': 19,
 'day': 3}

response = requests.post(url, json=data).json()


print(response)