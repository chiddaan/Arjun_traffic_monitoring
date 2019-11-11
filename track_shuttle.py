import json
import requests

buses_url = "http://uc.doublemap.com/map/v2/buses"
routes_url = "http://uc.doublemap.com/map/v2/routes"
stops_url = "http://uc.doublemap.com/map/v2/stops"

stops_on_north_route = requests.get(url= routes_url, params={'id':39})
stops_on_north_route_data = json.loads(stops_on_north_route.text)['stops']
# print(stops_on_north_route_data)

# PARAMS = {'route':39}
all_stops_request = requests.get(url= stops_url)
all_stops = json.loads(all_stops_request.text)
# for stop in all_stops:
#    if stop['id'] in stops_on_north_route_data:
#        print (stop)
# print(all_stops[0])



buses = requests.get(url=buses_url)
buses_data = json.loads(buses.text)
for bus in buses_data:
    if bus['route'] is 39:
        destination_stop = 106
        last_stop = bus['lastStop']
        index_of_last_stop = stops_on_north_route_data.index(last_stop)
        index_of_destination_stop = stops_on_north_route_data.index(destination_stop)
        difference = index_of_destination_stop - index_of_last_stop
        if difference > 0:
            print("The shuttle is {} stop(s) away".format(difference))
        elif difference == 0:
            print("The shuttle just left your stop")
        elif difference < 0:
            print("The shuttle is {} stop(s) ahead.".format(abs(difference)))

        # print(stops_on_north_route_data.index(last_stop))
        # print("last stop: {}".format(bus['lastStop']))
        # print("Heading: {}".format(bus['heading']))


# AIzaSyBB0wz1C6PqzYCl1u-aur-uMgsURdIAHqU
#
# https://maps.googleapis.com/maps/api/distancematrix/json?origins=Cincinnati,OH&destinations=Chicago,%20IL&departure_time=now&key=AIzaSyBB0wz1C6PqzYCl1u-aur-uMgsURdIAHqU
#
# destination latitude	39.143711
# longitude	-84.520523