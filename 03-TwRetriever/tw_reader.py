#!/usr/bin/python3 -tt
'''------------------------------------- >Last Modified on Thu, 23 Feb 2017< '''
from pymongo import MongoClient
from datetime import datetime
from pytz import timezone

n_tweets = 10 ## how many tweets are shown

client = MongoClient('mongodb://localhost:27017')
db = client.Tweets ## Database object (tweets)
collection = db.DataScience ## Collection object (DataScience)
tweets = collection.find().sort('id_str',direction=-1).limit(n_tweets) ## cursor

cnt = 0
for tweet in reversed([x for x in tweets]):
    cnt += 1
    id_str = tweet['id_str']
    when = datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S %z %Y')
    when = when.astimezone(timezone('Europe/Berlin')) ## UTC to CET

    who = tweet['user']['name'] ## user name
    what = tweet['text'] ## tweet text

    print('-----------------------------------------------',cnt)
    print(id_str,when.strftime('%d.%m. %H:%M'),who,sep='\t')
    print(what)
