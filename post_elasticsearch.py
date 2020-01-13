import requests
import json
import time
import pprint
from elasticsearch import Elasticsearch
from google_places import GooglePlaces

res = requests.get('http://localhost:9200')
print(res.content)
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
api = GooglePlaces("AIzaSyClBdGIIIoAaw34W_AIODpkcpDXs0EKpRc")
places = api.search_places_by_coordinate("49.280823, -123.121735", "50000", "bank", "RBC Royal Bank")
fields = ['name', 'formatted_address', 'international_phone_number', 'website', 'rating', 'review']
print(places)
for place in places:
    details = api.get_place_details(place['place_id'], fields)
    print("===================PLACE===================")
    pprint.pprint(details)
    print("==================REWIEVS==================")
    reviews = details['result']['reviews']
    for review in reviews:
        try:
            body = review
            body["website"] = details['result']['website']
            body["name"] = details['result']['name']
            body["formatted_address"] = details['result']['formatted_address']
            body["international_phone_number"] = details['result']['international_phone_number']
            pprint.pprint(body)
            id = review['author_url']+str(body['time'])
            print(id)
            es.create(index = "rbc", id = id, body = body)
        except KeyError:
            print('key error')
            continue