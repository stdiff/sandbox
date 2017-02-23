#!/usr/bin/python3 -tt

from datetime import datetime
import logging
import sys

from configparser import ConfigParser
from requests_oauthlib import OAuth1Session
from pymongo import MongoClient

config = ConfigParser()
config.read(sys.argv[1])

##############################################################################

logging.basicConfig(filename=config['Config']['LogFile'],
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S') ### Logger
logging.info('-------------------------- Start tw_retriever.py')

############################################################## Retrieve Tweets 

ua = OAuth1Session(config['Key']['ConsumerKey'],
                   client_secret=config['Key']['ConsumerSecret'],
                   resource_owner_key=config['Key']['AccessToken'],
                   resource_owner_secret=config['Key']['AccessTokenSecret'])

url = 'https://api.twitter.com/1.1/lists/statuses.json?list_id=%s' % \
      config['Config']['ListId']
response = ua.get(url)

try:
    tweets = response.json()
    logging.info('Retrieved %s tweets' % len(tweets))
except:
    logging.error('Got a non-JSON object')
    sys.exit()

################################################################# Store Tweets

client = MongoClient(config['Config']['MongoClient'])
db = client.Tweets ## Database object (Tweets)
collection = db.DataScience ## Collection object (DataScience)

cnt = 0
for tweet in tweets:
    id_str = tweet['id_str']
    dt = datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S %z %Y')
    tweet['timestamp'] = dt.timestamp()

    ## check if the tweet has already been stored
    if collection.find_one({'id_str': id_str}):
        continue

    try:
        collection.insert_one(tweet)
        cnt += 1
    except:
        logging.error('The insert failed: id_str=%s' % id_str)

logging.info('Stored %s tweets' % cnt)
