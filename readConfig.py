import configparser
import io
import datetime
import time


config = configparser.ConfigParser()
config.read('config.ini')
print(config['DEFAULT']['gpu'])


print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))