# Twitter Retreiver

To Fetch tweets in a list and store them on a MongoDB.

Before using this script, you need to register
[a new application](https://apps.twitter.com/)
and obtain Keys and AccessToken in advance.

## tw_retriever

	bash$ tw_retriever config.ini

This is the main script. It fetches new tweets in a specified list and store
them on a MongoDB.

- See below for `config.ini`. 
- We are able to specify only one list.
- The name of the collection is `DataScience`, you can change it directly 
  in the source code.

## config.ini

	[Config]
	ListId = 
	LogFile = 
	MongoClient = mongodb://localhost:27017

	[Key]
	ConsumerKey = 
	ConsumerSecret = 
	AccessToken = 
	AccessTokenSecret = 

- `ListID` : can be obtained via [GET lists/list](https://dev.twitter.com/rest/reference/get/lists/list).
- `LogFile` : log file for the script
- `MongoClient` : MongoDB URI
- `ConsumerKey`, `ConsumerSecret` : client key/secret pair 
- `AccessToken`, `AccessTokenSecret` : token credentials

## tw_reader

Show the recent 10 tweets from the MongoDB. 

In the source code you may directly want to change 

- the number of tweets to show (`n_tweets`),
- MongoDB URI (`client`), and 
- the name of Collection object (`collection`).

## Documentation

- [REST API](https://dev.twitter.com/rest/public)
- [Requests-OAuthlib](https://github.com/requests/requests-oauthlib)
- [PyMongo](https://api.mongodb.com/python/current/)




