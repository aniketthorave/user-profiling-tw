#!/usr/bin/env python
# encoding: utf-8
import re
import sys
import tweepy  # https://github.com/tweepy/tweepy
from nltk.corpus import stopwords
import textmining
import glob

# Twitter API credentials
consumer_key = "x1ZcTUl6KV6lxKkgcb51j91lI"
consumer_secret = "VFXPt4Gk6EwX4KbomR1HIAd9ObHjiWi5yMk5zyHbmjDPX07nMA"
access_key = "865138923282018304-VUvrEYMlCQRtRNX6OJYIFmZDTzRB5Yw"
access_secret = "QFYPrNqppR9gs2oWsVabv3TJpQPY6K5RcvyT2u4jih9Cg"

def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name,count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print "getting tweets before %s" % (oldest)

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print "...%s tweets downloaded so far" % (len(alltweets))

    # transform the tweepy tweets into a 2D array that will populate the csv
    #outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]


        # for tweet in alltweets:
    f = open('testfile.txt', 'w+')
    # x = f.read()
    # y = x
    # print(y)
    for tweet in alltweets:
        f.write(str(tweet.text))
        f.write('\n')
    f.close()
    tweets = (tweet.text)
    return tweets
    pass

# to encode the data
reload(sys)
sys.setdefaultencoding('utf-8')
if __name__ == '__main__':
    # pass in the username of the account you want to download
    get_all_tweets("tusharkute")


# stop word preprocessing
stop_words = set(stopwords.words('english'))
file1 = open("testfile.txt",'r')
line = file1.read()# Use this to read file content as a stream:
words = line.split()
with open('filteredtext.txt','w') as f:
    f.write(' ')
    f.close()
for r in words:
    if not r in stop_words:
        appendFile = open('filteredtext.txt','a')
        appendFile.write(" "+r)
        #appendFile.write('\n')
        appendFile.close()
with open('filteredtext.txt','r')as f1:
    line = f1.read()
    #print type(line)
    result = re.sub(r"http\S+", "", line)
    with open('result.txt','w')as f2:
        f2.write(result)
        f2.close()

#Document term Frequency matrix

tdm = textmining.TermDocumentMatrix()

files = glob.glob("result.txt")
for f in files:
    content = open(f).read()
    content = content.replace('\n', ' \n')
    tdm.add_doc(content)
    tdm.write_csv('matrix1.csv', cutoff=1)
