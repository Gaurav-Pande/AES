# -*- coding: utf-8 -*-
"""feature_extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1swVyQ8Uru2bC1-aTRi0GaNeep4qnssKp
"""
from requirements import *
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

def gender_summary(essays, gender_label, mcount, fcount, ecount, avg_size):
  for idx in range(len(essays)):
    if gender_label[idx] == 0:
      mcount += 1
    elif gender_label[idx] == 1:
      fcount += 1
    avg_size += len(essays[idx])
  ecount += len(essays)
  return mcount, fcount, ecount, avg_size

def age_summary(essays, age, age_label, c13, c14, c15, c16, c1, c2):
  for i in range(len(age)):
    if age[i] == 13:
      c13 += 1
    elif age[i] == 14:
      c14 += 1
    elif age[i] == 15:
      c15 += 1
    elif age[i] == 16:
      c16 += 1
    
    if age_label[i] == 1:
      c1 += 1
    elif age_label[i] == 2:
      c2 += 1
  return c13, c14, c15, c16, c1, c2

def split_in_sets(data):
    essay_sets = []
    min_scores = []
    max_scores = []
    for s in range(1,9):
        essay_set = data[data["essay_set"] == s]
        essay_set.dropna(axis=1, inplace=True)
        n, d = essay_set.shape
        set_scores = essay_set["domain1_score"]
        print ("Set", s, ": Essays = ", n , "\t Attributes = ", d)
        min_scores.append(set_scores.min())
        max_scores.append(set_scores.max())
        essay_sets.append(essay_set)
    return (essay_sets, min_scores, max_scores)

def possessives_features(posts):
  total_dict = {}
  count_my = {}
  count_others = {}

  index = 0

  for post in posts:
    tag = 0
    for word in post:
      if tag == 1 and word.lower() not in total_dict:
        total_dict[word.lower()] = index
        count_my[word.lower()] = 1
        count_others[word.lower()] = 0
        index += 1
      elif tag == 1 and word.lower() in total_dict:
        count_my[word.lower()] += 1
      elif tag == 2 and word.lower() not in total_dict:
        total_dict[word.lower()] = index
        count_others[word.lower()] = 1
        count_my[word.lower()] = 0
        index += 1
      elif tag == 2 and word.lower() in total_dict:
        count_others[word.lower()] += 1

      if word.lower() == 'my':
        tag = 1
      elif word.lower() == 'theirs' or word.lower() == 'his' or word.lower() == 'hers' or word.lower() == 'yours':
        tag = 2
      else:
        tag = 0
  
  my_feature = []
  other_feature = []
  dict_len = len(total_dict)
  #print('dict len is',dict_len, 'index', index)

  for post in posts:
    tag = 0
    feature1 = np.zeros(dict_len)
    feature2 = np.zeros(dict_len)
    for word in post:
      if tag == 1:
        idx = total_dict[word.lower()]
        feature1[idx] += 1
      elif tag == 2:
        idx = total_dict[word.lower()]
        feature2[idx] += 1

      if word.lower() == 'my':
        tag = 1
      elif word.lower() == 'theirs' or word.lower() == 'his' or word.lower() == 'hers' or word.lower() == 'yours':
        tag = 2
      else:
        tag = 0
    my_feature.append(feature1)
    other_feature.append(feature2)
  return my_feature, other_feature, total_dict

nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
tagset = nltk.load('help/tagsets/upenn_tagset.pickle')
no_tags = len(tagset.keys())
index = 0
tagdict = {}
for tag in tagset.keys():
  tagdict[tag] = index
  index = index+1

def POS(essays):
  pos_features = []
  for post in essays:
    pos = nltk.pos_tag(post)
    feature = np.zeros(no_tags)
    for elem in pos:
      tag = elem[1]
      index = tagdict[tag]
      feature[index] += 1
    pos_features.append(feature)
  return pos_features

#f:nn, jj, vbp, rb, vbd, cc
#m:nnp, dt
def custom(pos_features, tagdict):
  c_features = []
  w= [9,10,5,8,6,14,8]

  for feature in pos_features:
    i0 = tagdict['NN']
    i1 = tagdict['JJ']
    i2 = tagdict['VBP']
    i3 = tagdict['VBD']
    i4 = tagdict['CC']

    i5 = tagdict['NNP']
    i6 = tagdict['DT']

    num = w[0]*feature[i0]+w[1]*feature[i1]+w[2]*feature[i2]+w[3]*feature[i3]+w[4]*feature[i4]-w[5]*feature[i5]-w[6]*feature[i6]
    c_features.append(num)
  return c_features

### Reference: https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
def f(pos_features):
  f_features = []
  for feature in pos_features:
    ##NN, NNS, NNP, NNPS
    i0 = tagdict['NN']
    i1 = tagdict['NNS']
    i2 = tagdict['NNP']
    i3 = tagdict['NNPS']
    noun = feature[i0]+feature[i1]+feature[i2]+feature[i3]

    #JJ,JJR,JJS
    i4 = tagdict['JJ']
    i5 = tagdict['JJR']
    i6 = tagdict['JJS']
    adj = feature[i4]+feature[i5]+feature[i6]

    #IN
    i7 = tagdict['IN']
    prep = feature[i7]

    #DT,WDT, PDT(?)
    i8 = tagdict['DT']
    i9 = tagdict['WDT']
    art = feature[i8]+feature[i9]

    #PRP, PRP$
    i10 = tagdict['PRP']
    i11 = tagdict['PRP$']
    pron = feature[i10]+feature[i11]

    #VB,VBD, VBG, VBN, VBP, VBZ
    i12 = tagdict['VB']
    i13 = tagdict['VBD']
    i14 = tagdict['VBN']
    i15 = tagdict['VBP']
    i16 = tagdict['VBZ']
    verb = feature[i12]+feature[i13]+feature[i14]+feature[i15]+feature[i16]

    #RB,RBR,RBS
    i17 = tagdict['RB']
    i18 = tagdict['RBR']
    i19 = tagdict['RBS']
    adv =  feature[i17]+feature[i18]+feature[i19]

    #UH
    i20 = tagdict['UH']
    inter = feature[i20]
    num = 0.5*((noun+adj+prep+art)-(verb+adv+inter)+100)
    f_features.append(num)
  return f_features

index = 0
tagseq = {}
for tag1 in tagset.keys():
  for tag2 in tagset.keys():
    tog = tag1+" "+tag2
    if tog not in tagseq:
      tagseq[tog] = index
      index = index+1

def pos_seq(essays):
  pos_seq_features = []
  tagseqlen = len(tagseq)
  for post in essays:
    pos = nltk.pos_tag(post)
    feature = np.zeros(tagseqlen)
    for i in range(len(pos)-1):
      tag1 = pos[i]
      tag2 = pos[i+1]
      tag1 = tag1[1]
      tag2 = tag2[1]
      tog = tag1+" "+tag2
      index = tagseq[tog]
      feature[index] += 1
    pos_seq_features.append(feature)
  return pos_seq_features
  

def ngram_train_char(data, num, count_vect):
  if count_vect == []:
    count_vect = CountVectorizer(stop_words='english', analyzer='char', ngram_range=(num, num))
  ngram_counts = count_vect.fit_transform(data)
  return ngram_counts, count_vect

def ngram_test_char(data, model):
  ngram_counts = model.transform(data)
  return ngram_counts

def ngram_train(data, num, count_vect):
  if count_vect == []:
    count_vect = CountVectorizer(stop_words='english', analyzer='word', ngram_range=(num, num))
  ngram_counts = count_vect.fit_transform(data)
  return ngram_counts, count_vect

def ngram_test(data, model):
  ngram_counts = model.transform(data)
  return ngram_counts

nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
tagset = nltk.load('help/tagsets/upenn_tagset.pickle')
no_tags = len(tagset.keys())

def pos_cvt(data):
  pos_essays = []
  for post in data:
    pos = nltk.pos_tag(post)
    new_pos = [elem[1] for elem in pos]
    pos_essays.append(' '.join(new_pos))
  return pos_essays

def penn_to_wn(tag):
    if tag[1][0] == 'J':
        return wn.ADJ
    elif tag[1][0] =='N':
        return wn.NOUN
    elif tag[1][0] == 'R':
        return wn.ADV
    elif tag[1][0] == 'V':
        return wn.VERB
    return None

nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
tagset = nltk.load('help/tagsets/upenn_tagset.pickle')



lemmatizer = WordNetLemmatizer()

def sentiment_features(essays):
  f_pos_score = 0
  f_neg_score = 0
  f_obj_score = 0
  f_count = 0

  m_pos_score = 0
  m_neg_score = 0
  m_obj_score = 0
  m_count = 0

  mc = 0
  fc = 0

  total_feat = []
  for post_id in range(len(essays)):
    post = essays[post_id]
  #g = gender_label[post_id]

    pos = nltk.pos_tag(post)
    feat = np.zeros(3)

    for idx in range(len(pos)):
      tag = pos[idx]
      word = post[idx]
      wn_tag = penn_to_wn(tag)
      if wn_tag == None:
        continue
      lemma = lemmatizer.lemmatize(word, pos=wn_tag)
      if lemma == []:
        continue
      synsets = wn.synsets(word, pos=wn_tag)
      if synsets == []:
        continue

      synset = synsets[0] #most common
      swn_synset = swn.senti_synset(synset.name())
      pos_score = swn_synset.pos_score()
      neg_score = swn_synset.neg_score()
      obj_score = swn_synset.obj_score()
      feat[0] += pos_score
      feat[1] += neg_score
      feat[2] += obj_score
    total_feat.append(feat)
  return total_feat

import pyphen
dic = pyphen.Pyphen(lang='en')

def extract_features(essays, age):
  features = []
  age_label = []
  for e in essays:
    n_words = len(e)

    if e == []:
      continue
    complex_words = 0
    syllables = 0
    n_chars = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    n6 = 0
    n7 = 0
    n8 = 0

    for word in e:
      syl = dic.inserted(word)
      syllables += syl.count('-') 
      sc = syl.count('-')
      if sc > 2:
        complex_words += 1
      if sc == 1:
        n1 += 1
      if sc == 2:
        n2 += 1
      if sc == 3:
        n3 += 1
      if sc == 4:
        n4 += 1
      if sc == 5:
        n5 += 1
      if sc == 6:
        n6 += 1
      if sc == 7:
        n7 += 1
      if sc == 8:
        n8 += 1
      n_chars += len(word)
    f = [n_chars/n_words, n_words, complex_words/n_words, syllables/n_words, n1/n_words, n2/n_words, n3/n_words, n4/n_words, n5/n_words, n6/n_words, n7/n_words, n8/n_words]
    features.append(f)
    age_label.append(age[essays.index(e)])
  return features, age_label

def pentel_features(essays):
  features = []
  age_label = []
  for e in essays:
    n_words = len(e)

    if e == []:
      continue
    complex_words = 0
    syllables = 0
    n_chars = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    n6 = 0
    n7 = 0
    n8 = 0

    for word in e:
      syl = dic.inserted(word)
      syllables += syl.count('-') 
      sc = syl.count('-')
      if sc > 2:
        complex_words += 1
      if sc == 1:
        n1 += 1
      if sc == 2:
        n2 += 1
      if sc == 3:
        n3 += 1
      if sc == 4:
        n4 += 1
      if sc == 5:
        n5 += 1
      if sc == 6:
        n6 += 1
      if sc == 7:
        n7 += 1
      if sc == 8:
        n8 += 1
      n_chars += len(word)
    f = [n_chars/n_words, n_words, complex_words/n_words, syllables/n_words, n1/n_words, n2/n_words, n3/n_words, n4/n_words, n5/n_words, n6/n_words, n7/n_words, n8/n_words]
    features.append(f)
  return features
