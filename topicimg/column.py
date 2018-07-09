#!/usr/bin/env python
# encoding: utf-8
import string
import sys
from itertools import chain
import gensim
import tweepy  # https://github.com/tweepy/tweepy
from gensim import corpora,models
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
        #print outtweets
        #f=open('testfile.txt','w')
        #f.write(str(outtweets))
        #f.close()
    return outtweets
    pass
# to encode the data
reload(sys)
sys.setdefaultencoding('utf-8')

#pass in the username of the account you want to download

doc_complete=get_all_tweets('tusharkute')


#Cleaning and Preprocessing

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    #print normalized
    return normalized



doc_clean_l = [clean(doc).split() for doc in doc_complete]
doc_clean = doc_clean_l[:20]

#print doc_clean

#Preparing Document-Term Matrix
# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
file3=open('DTFM.txt','w')
file3.write(str(doc_term_matrix))
file3.close()


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

SOME_FIXED_SEED = 42

# before training/inference:
np.random.seed(SOME_FIXED_SEED)

SOME_FIXED_SEED = 42

# before training/inference:
np.random.seed(SOME_FIXED_SEED)

mm = doc_term_matrix
# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=3, \
                               update_every=1, chunksize=10000, passes=1)

# Prints the topics.
#topic to word distribution
for top in lda.print_topics():
  print top
print

# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]
from itertools import chain
# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
scores = list(chain(*[[score for topic_id,score in topic] \
                     for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)

# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
#scores = []
#for doc in lda_corpus:
#    for topic in doc:
#        for topic_id, score in topic:
#            scores.append(score)
#threshold = sum(scores)/len(scores)

print threshold
print

newtexts = doc_clean

cluster1 = [j for i,j in zip(lda_corpus,newtexts) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,newtexts) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,newtexts) if i[2][1] > threshold]

print cluster1
print cluster2
print cluster3



# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(mm, num_topics=3, id2word = dictionary, passes=50)


lda=ldamodel.print_topics(num_topics=3, num_words=10)
#type(ldamodel)
f=open('LDA.txt','w')
f.write(str(lda))
f.close()

#to convert list of tuple into column
# When reading your file, call ast.literal_eval.
import ast
with open('LDA.txt') as f:
    data = ast.literal_eval(f.read())

import re
import pandas as pd
d = {}
for i, y in data:
    d['topic {}'.format(i)] = re.findall('"(.*?)"', y)
df = pd.DataFrame(d)
df.to_csv('column.csv')


'''from collections import defaultdict
from nltk.corpus import wordnet as wn

# Loading the Wordnet domains.
domain2synsets = defaultdict(list)
synset2domains = defaultdict(list)
for i in open('wn-domains-3.2-20070223', 'r'):
    ssid, doms = i.strip().split('\t')
    doms = doms.split()
    synset2domains[ssid] = doms
    for d in doms:
        domain2synsets[d].append(ssid)

# Gets domains given synset.
for ss in wn.all_synsets():
    ssid = str(ss.offset).zfill(8) + "-" + ss.pos()
    if synset2domains[ssid]: # not all synsets are in WordNet Domain.
        print ss, ssid, synset2domains[ssid]

# Gets synsets given domain.
for dom in sorted(domain2synsets):
    print dom, domain2synsets[dom][:3]'''


sys1 =[]
#for the full words distributions for each topic and all topics in your lda model.
for i in ldamodel.show_topics(formatted=False,num_topics=ldamodel.num_topics,num_words=len(ldamodel.id2word)):
    sys1.append(i)
f2=open('num_words.txt','w')
f2.write(str(sys1))
f2.close()
    # print i

#top n topic
topicid = input("\n enter topic id: ")
topn = input("\n enter top n words: ")
for i in ldamodel.show_topic(topicid,topn):
    print i




'''result = []
import re
for i in range(len(texts)):
    result.append(re.sub(r"http\S+", "", str(texts[i])))



#!/usr/bin/env python
# encoding: utf-8
import string
import sys
from gensim import corpora, models, similarities
from itertools import chain
import gensim
import tweepy  # https://github.com/tweepy/tweepy
from gensim import corpora
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
        #print outtweets
        #f=open('testfile.txt','w')
        #f.write(str(outtweets))
        #f.close()
    return outtweets
    pass
# to encode the data
reload(sys)
sys.setdefaultencoding('utf-8')

#pass in the username of the account you want to download

doc_complete=get_all_tweets('tusharkute')
documents = doc_complete[:10]
print documents

from gensim import corpora, models, similarities
from itertools import chain

""" DEMO
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]"""

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

import re
for i in range(len(texts)):
    x = texts[i]

    re.findall(r"#(\w+)", str(x[0]))

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]

# Create Dictionary.
id2word = corpora.Dictionary(texts)
# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=3, \
                               update_every=1, chunksize=10000, passes=1)

# Prints the topics.
for top in lda.print_topics():
  print top
print

# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]

# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print threshold
print

cluster1 = [j for i,j in zip(lda_corpus,documents) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,documents) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,documents) if i[2][1] > threshold]

print cluster1
print cluster2'''
#print cluster3