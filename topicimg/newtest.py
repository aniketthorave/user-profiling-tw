#!/usr/bin/env python
# encoding: utf-8
import string
import sys
from gensim import corpora, models, similarities
from itertools import chain
import gensim.models.ldamulticore
import tweepy  # https://github.com/tweepy/tweepy
from gensim import corpora
from gensim.models import LdaMulticore, ldamodel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
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
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

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
        outtweets = [tweet.text for tweet in alltweets]

    return outtweets
    pass
# to encode the data
reload(sys)
sys.setdefaultencoding('utf-8')

#pass in the username of the account you want to download

doc_complete=get_all_tweets('@imVkohli')
documents = doc_complete[:10]
#print documents
f=open('tweet.txt','w')
f.write(str(documents))
f.close()

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    #print normalized
    return normalized


def printloop(data, msg):
    print msg.center(20,"_")
    for i in range(len(data)):
        print data[i]
    print "\n"


#stoplist = set('for a of the and to in'.split())
#texts = [[word for word in document.lower().split() if word not in stoplist]
#         for document in documents]
texts = [clean(doc).split() for doc in documents]
# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set   (word for word in set(all_tokens) if all_tokens.count(word) == 1)

newtexts = [[word for word in text if word not in tokens_once] for text in texts]

# Create Dictionary.
id2word = corpora.Dictionary(newtexts)

# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in newtexts]

#print "Print Document Term Matrix \n"
printloop(mm, "Document Term Matrix")

SOME_FIXED_SEED = 42

# before training/inference:
np.random.seed(SOME_FIXED_SEED)

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
num_topics = input("Enter number of topics you want:")
passes = input("Enter passes you want: ")
# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=num_topics, update_every=1, chunksize=10000, passes=passes)

# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]

# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
scores = list(chain(*[[score for topic_id,score in topic] \
                     for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print "\n Threshold \n"
print threshold

cluster1 = [j for i,j in zip(lda_corpus,newtexts) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,newtexts) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,newtexts) if i[2][1] > threshold]

print "\n Print Randomly assign each word in the document \n"

printloop(cluster1, "cluster 1")
printloop(cluster2, "cluster 2")
printloop(cluster3, "cluster 3")

#topic to Document distribution
#lda = LdaMulticore(mm, id2word=id2word, num_topics=3)  # train model
#print "print Topic to Document Distribution \n"
lda2 = LdaMulticore(corpus=mm, num_topics=lda.num_topics, id2word=id2word, alpha=1e-5, eta=5e-1,
              minimum_probability=0.0) #train model
Document_id = input("Enter Document id: ")
T_W_D = (lda2[mm[Document_id]]) # get topic probability distribution for a document
#lda.update(mm2) # update the LDA model with additional documents

printloop(T_W_D, "Topic to Document Distribution ")

print "\n Print Topic to Word Distribution \n"
#topic to word distribution
for top in lda.print_topics():
  print top


#print "Print full word Distribution for each topic \n"
sys1 =[]
#for the full words distributions for each topic and all topics in your lda model.
for i in lda.show_topics(formatted=False,num_topics=lda.num_topics,num_words=len(lda.id2word)):
    sys1.append(i)

printloop(sys1, "\n full word Distribution for each topic \n")


#num_words = input("enter number of words you want:")
lda1=lda.print_topics(num_topics=lda.num_topics, num_words=10)
f=open('LDA.txt','w')
for i in range(len(lda1)):
    f.write(str(lda1[i]))
    f.write("\n")
f.close()

#top n topic
print "\n print Top n Topic and Top n Words From respective topic \n"
topicid = input("\n enter topic id: ")
topn = input("\n enter top n words: ")
for i in lda.show_topic(topicid,topn):
    print i

#to convert list of tuple into column
# When reading your file, call ast.literal_eval.
'''import ast
with open('LDA.txt') as f:
    data = ast.literal_eval(f.read())'''

data = lda1
import re
import pandas as pd
d = {}
for i, y in data:
    d['topic {}'.format(i)] = re.findall('"(.*?)"', y)
df = pd.DataFrame(d)
df.to_csv('column.csv')
