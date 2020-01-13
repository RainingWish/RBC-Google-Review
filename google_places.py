import requests
import json
import time
import pprint
import elasticsearch


class GooglePlaces(object):
    def __init__(self, apiKey):
        super(GooglePlaces, self).__init__()
        self.apiKey = apiKey

    def search_places_by_coordinate(self, location, radius, types,name):
        endpoint_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        places = []
        params = {
            'location': location,
            'radius': radius,
            'types': types,
            'key': self.apiKey,
            'name': name,
        }
        res = requests.get(endpoint_url, params=params)
        print(res)
        results = json.loads(res.content)
        print(results)
        places.extend(results['results'])
        time.sleep(2)
        while "next_page_token" in results:
            params['pagetoken'] = results['next_page_token'],
            res = requests.get(endpoint_url, params=params)
            results = json.loads(res.content)
            places.extend(results['results'])
            time.sleep(2)
        return places

    def get_place_details(self, place_id, fields):
        endpoint_url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'placeid': place_id,
            'fields': ",".join(fields),
            'key': self.apiKey
        }
        res = requests.get(endpoint_url, params=params)
        print(res)
        place_details = json.loads(res.content)
        return place_details


if __name__ == '__main__':
    api = GooglePlaces("AIzaSyClBdGIIIoAaw34W_AIODpkcpDXs0EKpRc")
    places = api.search_places_by_coordinate("45.362944, -75.738690", "10000", "bank","RBC Royal Bank")
    fields = ['name', 'formatted_address', 'international_phone_number', 'website', 'rating', 'review']
    print(places)
    for place in places:
        details = api.get_place_details(place['place_id'], fields)
        print(details)
        try:
            website = details['result']['website']
        except KeyError:
            website = ""

        try:
            name = details['result']['name']
        except KeyError:
            name = ""

        try:
            address = details['result']['formatted_address']
        except KeyError:
            address = ""

        try:
            phone_number = details['result']['international_phone_number']
        except KeyError:
            phone_number = ""

        try:
            reviews = details['result']['reviews']
        except KeyError:
            reviews = []
        print("===================PLACE===================")
        print("Name:", name)
        print("Website:", website)
        print("Address:", address)
        print("Phone Number", phone_number)
        print("==================REWIEVS==================")
        for review in reviews:
            pprint.pprint(review)
            author_name = review['author_name']
            rating = review['rating']
            text = review['text']
            time = review['relative_time_description']
            profile_photo = review['profile_photo_url']
            time = review['time']
            author_url = review['author_url']

            print("-----------------------------------------")