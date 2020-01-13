import requests
import json
import time
import pprint
from elasticsearch import Elasticsearch
from google_places import GooglePlaces
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from nltk import FreqDist
import re
import spacy
import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#%matplotlib inline


# set up
res = requests.get('http://localhost:9200')
print(res.content)
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# plot function from https://www.analyticsvidhya.com/blog/2018/10/mining-online-reviews-topic-modeling-lda/
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms)
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()

# function to remove stopwords
def remove_stopwords(rev):
    stop_words = stopwords.words('english')
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


def lemmatization(texts, tags=['NOUN', 'ADJ']):  # filter noun and adjective
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

index = 'rbc'
body = {
    "from": 0, "size": 9999,
    "query": {
        "range" : {
            "rating" : {
                "gte" : 1,
                "lte" : 2
            }
        }
    }
}
raw_data = es.search(index=index, body=body)
df_data = raw_data['hits']['hits']
list_data = list()
for review in raw_data['hits']['hits']:
    #print(review)
    source_body = review['_source']
    source_body['id'] = review['_id']
    list_data.append(source_body)

df_data = pd.DataFrame(list_data)

# parse out untext
df_data['processed_text'] = df_data['text'].str.replace("[^a-zA-Z#]", " ")
# remove stop word
reviews = [remove_stopwords(r.split()) for r in df_data['processed_text']]
# remove short word
df_data['processed_text'] = df_data['processed_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
# lowercase
reviews = [r.lower() for r in reviews]

# remove common banking word
common_words = [
    'bank',
    'branch',
    'staff'
]
df_data['processed_text'] = df_data['processed_text'].apply(lambda x: ' '.join([w for w in x.split() if w in common_words]))

tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
reviews_2 = lemmatization(tokenized_reviews)
reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

df_data['text'] = reviews_3

# model
dictionary = corpora.Dictionary(reviews_2)
dictionary.filter_extremes(no_below=10, no_above=0.22, keep_n= 100000)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel
print('reviews_2',reviews_2)
print('reviews_3',reviews_3)

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=4, random_state=100,
                chunksize=1000, passes=40, update_every=1)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))




def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=doc_term_matrix, texts=df_data['text'])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

topic = pd.Series(df_dominant_topic['Dominant_Topic'])
contribution = pd.Series(df_dominant_topic['Topic_Perc_Contrib'])
key_words = pd.Series(df_dominant_topic['Keywords'])

df_data = pd.concat([df_data, topic,contribution,key_words], axis=1)

put_back_list = df_data.T.to_dict().values()
for row in put_back_list:
    row['review_type'] = 'negative'
    try:
        es.create(index="rbc_processed", id=row['id'], body=row)
        print(row)
    except:
        'error, next'
