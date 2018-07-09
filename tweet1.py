#!/usr/bin/env python
# encoding: utf-8
import re
import pandas as pd
import string
import sys
from itertools import chain
import gensim.models.ldamulticore
import sys
from itertools import chain
import gensim.models.ldamulticore
import numpy as np
import tweepy  # https://github.com/tweepy/tweepy
from gensim import corpora
from gensim import models
from gensim.models import LdaMulticore
import string
import re
import ast
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
# Twitter API credentials
consumer_key = "Enter your consumer_key"
consumer_secret = "Enter your Consumer_secret"
access_key = "Enter your access_key"
access_secret = "Enter access_secret"

reload(sys)
sys.setdefaultencoding('utf-8')


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


def fetch_tweet():
    doc_complete = get_all_tweets('tusharkute')
    documents = doc_complete[:10]
    # print documents
    f = open('tweet.txt', 'w')
    f.write(str(documents))
    f.close()
    s = pd.Series(documents)
    s.to_csv("newtweet.txt", index=False)
    return s

def main():
    while True:
        print "MENU"
        print "Enter 1 for download tweets"
        print "Enter 2 for preprocessing"
        print "Enter Zero to Exit"
        choice=input("Enter your choice:")
        if choice == 1:
            # to encode the data


            # pass in the username of the account you want to download
            s = fetch_tweet()
            print s

        elif choice == 2:
            cache_english_stopwords = stopwords.words('english')

            filename = 'tweet.txt'
            file = open(filename, 'rt')
            text = file.read()
            text = ast.literal_eval(text)
            file.close()
            text2 = list()
            for i in range(len(text)):
                text2.append(text[i].decode('unicode_escape').encode('ascii', 'ignore'))
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
            print(emoji_pattern.sub(r'', str(text2)))  # no emoji

            def tweet_clean(tweet):
                text2 = tweet
                # Remove tickers
                sent_no_tickers = re.sub(r'\$\w*', '', text2)
                print('No tickers:')
                print(sent_no_tickers)
                tw_tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
                temp_tw_list = tw_tknzr.tokenize(sent_no_tickers)
                print('Temp_list:')
                print(temp_tw_list)
                # Remove stopwords
                list_no_stopwords = [i for i in temp_tw_list if i.lower() not in cache_english_stopwords]
                print('No Stopwords:')
                print(list_no_stopwords)
                # Remove hyperlinks
                list_no_hyperlinks = [re.sub(r'https?:\/\/.*\/\w*', '', i) for i in list_no_stopwords]
                print('No hyperlinks:')
                print(list_no_hyperlinks)
                # Remove Punctuation and split 's, 't, 've with a space for filter
                list_no_punctuation = [re.sub(r'[' + string.punctuation + ']+', ' ', i) for i in list_no_hyperlinks]
                print('No punctuation:')
                print(list_no_punctuation)
                # Remove multiple whitespace
                new_sent = ' '.join(list_no_punctuation)
                # Remove any words with 2 or fewer letters
                filtered_list = tw_tknzr.tokenize(new_sent)
                list_filtered = [re.sub(r'^\w\w?$', '', i) for i in filtered_list]
                print('Clean list of words:')
                s1 = list_filtered
                print(s1)

                filtered_sent = ' '.join(s1)
                clean_sent = re.sub(r'\s\s+', ' ', filtered_sent)
                # Remove any whitespace at the front of the sentence
                clean_sent = clean_sent.lstrip(' ')
                print('Clean sentence:')
                print clean_sent
                return clean_sent
                # return list_filtered

            filter_tweet = list()
            for i in range(len(text2)):
                onetweet = tweet_clean(text2[i])
                tweetlist = onetweet.split()
                filter_tweet.append(tweetlist)
            '''f=open("clean_tweet.txt","w")
            f.write(str(filter_tweet))
            f.close()
            filename = 'clean_tweet.txt'
            file = open(filename, 'rt')
            text1 = file.read()
            file.close()'''

            def printloop(data, msg):
                print msg.center(20, "_")
                for i in range(len(data)):
                    print data[i]
                print "\n"

            id2word = corpora.Dictionary(filter_tweet)

            # Creates the Bag of Word corpus.
            mm = [id2word.doc2bow(text) for text in filter_tweet]
            # print "Print Document Term Matrix \n"
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
            lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=num_topics, update_every=1,
                                           chunksize=10000, passes=passes)

            # Assigns the topics to the documents in corpus
            lda_corpus = lda[mm]

            # Find the threshold, let's set the threshold to be 1/#clusters,
            # To prove that the threshold is sane, we average the sum of all probabilities:
            scores = list(chain(*[[score for topic_id, score in topic] \
                                  for topic in [doc for doc in lda_corpus]]))
            threshold = sum(scores) / len(scores)
            print "\n Threshold \n"
            print threshold

            cluster1 = [j for i, j in zip(lda_corpus, filter_tweet) if i[0][1] > threshold]
            cluster2 = [j for i, j in zip(lda_corpus, filter_tweet) if i[1][1] > threshold]
            cluster3 = [j for i, j in zip(lda_corpus, filter_tweet) if i[2][1] > threshold]

            print "\n Print Randomly assign each word in the document \n"

            printloop(cluster1, "cluster 1")
            printloop(cluster2, "cluster 2")
            printloop(cluster3, "cluster 3")

            # topic to Document distribution
            # lda = LdaMulticore(mm, id2word=id2word, num_topics=3)  # train model
            # print "print Topic to Document Distribution \n"
            lda2 = LdaMulticore(corpus=mm, num_topics=lda.num_topics, id2word=id2word, alpha=1e-5, eta=5e-1,
                                minimum_probability=0.0)  # train model
            Document_id = input("Enter Document id: ")
            T_W_D = (lda2[mm[Document_id]])  # get topic probability distribution for a document
            # lda.update(mm2) # update the LDA model with additional documents

            printloop(T_W_D, "Topic to Document Distribution ")

            print "\n Print Topic to Word Distribution \n"
            # topic to word distribution
            for top in lda.print_topics():
                print top

            # print "Print full word Distribution for each topic \n"
            sys1 = []
            # for the full words distributions for each topic and all topics in your lda model.
            for i in lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=len(lda.id2word)):
                sys1.append(i)

            printloop(sys1, "\n full word Distribution for each topic \n")

            # num_words = input("enter number of words you want:")
            lda1 = lda.print_topics(num_topics=lda.num_topics, num_words=10)
            f = open('LDA.txt', 'w')
            for i in range(len(lda1)):
                f.write(str(lda1[i]))
                f.write("\n")
            f.close()

            # top n topic
            print "\n print Top n Topic and Top n Words From respective topic \n"
            topicid = input("\n enter topic id: ")
            topn = input("\n enter top n words: ")
            for i in lda.show_topic(topicid, topn):
                print i

            # to convert list of tuple into column
            # When reading your file, call ast.literal_eval.
            '''import ast
            with open('LDA.txt') as f:
                data = ast.literal_eval(f.read())'''

            data = lda1

            d = {}
            for i, y in data:
                d['topic {}'.format(i)] = re.findall('"(.*?)"', y)
            df = pd.DataFrame(d)
            df.to_csv('column.csv')
        elif choice==0:
            exit(0)
main()
