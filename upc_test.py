#!/usr/bin/env python
# encoding: utf-8
import codecs
import sys

import gensim
import tweepy  # https://github.com/tweepy/tweepy
from gensim import corpora
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

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
    f = codecs.open('testfile.txt', 'w+',encoding='utf-8',errors='ignore')
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
    get_all_tweets("tejasprawal")

#stopped_tokens = unicode(str, errors='replace')

tokenizer = RegexpTokenizer(r'\w+')
file1 = codecs.open("testfile.txt",'r',encoding='utf-8',errors='ignore')
line = file1.read()# Use this to read file content as a stream:
raw = line.lower()
tokens = tokenizer.tokenize(raw)
print(tokens)

#preprocessing on words

texts = []

en_stop = get_stop_words('en')
stopped_tokens = [i for i in tokens if not i in en_stop]
print stopped_tokens

p_stemmer = PorterStemmer()
for i in line:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
file3=open('DTFM.txt','w')
file3.write(str(corpus))
file3.close()
#print corpus
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)
file4=open('LDA.txt','w')
file4.write(str(ldamodel))
file4.close()
#print ldamodel
exit(0)