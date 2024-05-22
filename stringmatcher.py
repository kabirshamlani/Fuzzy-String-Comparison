from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
import torch
from difflib import SequenceMatcher
import jellyfish
from thefuzz import fuzz
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
import nltk
# import index.html
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize the stemmer
stemmer = PorterStemmer()
def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    similarity = dot_product / (norm_u * norm_v)

    return similarity



# from nltk.corpus import stopwords
# english_stopwords = set(stopwords.words('english'))


import math
def split_sentence(sentence, n):
    words = sentence.split()
    num_words = len(words)
    new_words=[]
    if(num_words<=3):
      return [sentence]
    else:
      for i in range(len(words)-n+1):
        new_words.append(" ".join(words[i:i+n]))

    return new_words



class checksimilarity:

  def __init__(self):
    pass

  def EmbeddingSimilarity(self,x1,x2):
    SentTransformer= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_1= SentTransformer.encode(x1)
    embedding_2 = SentTransformer.encode(x2)
    # print(x1,x2)
    sim=[]
    for i in range(len(embedding_1)):
      sim.append(cosine_similarity(embedding_1[i],embedding_2[i]))
    return sim

  def tolower(self,s1,s2):
    return s1.lower(),s2.lower()

  def stemwords(self,x1,x2):
    for i in range(len(x1)):
      x1[i]=stemmer.stem(x1[i])
      x2[i]=stemmer.stem(x2[i])
    return x1,x2

  def SequenceMatcher(self,s1 ,s2):
    s = SequenceMatcher(None, s1, s2)
    return s.ratio()

  def levenshtein_distance(self,s1,s2):
    return jellyfish.levenshtein_distance(s1,s2)

  def jaro_distance(self,s1,s2):
    return jellyfish.jaro_similarity(s1,s2)

  def damerau_levenshtein_distance(self,s1,s2):
    return jellyfish.damerau_levenshtein_distance(s1,s2)

  def jaro_winkler_similarity(self,s1,s2):
    return jellyfish.jaro_winkler_similarity(s1,s2)

  def fuzz_ratio(self,s1,s2):
    return fuzz.ratio(s1,s2)/100

  def fuzz_partial_ratio(self,s1,s2):
    return fuzz.partial_ratio(s1,s2)

  def fuzz_token_set_ratio(self,s1,s2):
    return fuzz.token_set_ratio(s1,s2)

  def fuzz_token_sort_ratio(self,s1,s2):
    return fuzz.token_set_ratio(s1,s2)

  def normalize_scores(self,scores,s1,s2):
    scores[1] = 1 - min(1, scores[1] / max(1,max(len(s1), len(s2))))
    scores[4] = 1 - min(1, scores[4] / max(1,max(len(s1), len(s2))))
    scores[7]/=100
    scores[8]/=100
    scores[9]/=100

    return scores

  def simple_average(self, scores):
    return np.mean(scores)

  def mean_ensemble_model(self,l):
    scores = self.EnsembleModel(l)
    return scores

  def weighted_average(self, scores, weights):
    return np.average(scores, weights=weights)

  def ensemble_model_weighted(self,l):
      x1=[]
      x2=[]
      for i in range(len(l)):
        x1.append(l[i][0])
        x2.append(l[i][1])

      scores = self.EnsembleModel(x1, x2)
      weights = [0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
      return self.weighted_average(scores, weights)

  def EnsembleModel(self,l):
    x1=[]
    x2=[]
    for i in range(len(l)):
      x1.append(l[i][0])
      x2.append(l[i][1])

    res0=self.EmbeddingSimilarity(x1,x2)
    x1,x2=self.stemwords(x1,x2)
    scores=[]
    for i in range(len(x1)):
      s1,s2=x1[i],x2[i]
      res1=res0[i]
      res2=self.damerau_levenshtein_distance(s1,s2)
      res3=self.fuzz_ratio(s1,s2)
      res4=self.jaro_distance(s1,s2)
      res5=self.levenshtein_distance(s1,s2)
      res6=self.SequenceMatcher(s1,s2)
      res7=self.jaro_winkler_similarity(s1,s2)
      res8=self.fuzz_partial_ratio(s1,s2)
      res9=self.fuzz_token_set_ratio(s1,s2)
      res10=self.fuzz_token_sort_ratio(s1,s2)
      res=[res1,res2,res3,res4,res5,res6,res7,res8,res9,res10]
      # print(res)
      normalized_scores = self.normalize_scores(res,s1,s2)
      avg=self.simple_average(normalized_scores)
      if(s1 in s2 or s2 in s1):
        scores.append(1)
      else:
        if(avg==None):
          scores.append(avg)
        elif(avg<0.65):
          if(normalized_scores[0]>=0.75):
            scores.append(normalized_scores[0])
          else:
            scores.append(avg)
        else:
          scores.append(avg)

    return scores


#   def WordnetSynsetsimilarity(self,s1,s2):
#     try:
#       cs1=wn.synsets(s1)
#       if(len(cs1)==0):
#         s1,s2=s2,s1

#       syn1=wn.synset(f"{s1}.n.01")
#       new_words=[str(lemma.name()) for lemma in syn1.lemmas()]
#       best=0
#       best_word=""
#       for i in new_words:
#         sim=self.mean_ensemble_model(i,s2)
#         if(sim>best):
#           best_word=i
#         if(best>=0.8):
#           return best
#         best=max(best,sim)
#       return best
#     except:
#       pass



import re
def remove_urls(text):

    suffix_pattern = r'\b(\.[a-z]{2,3}(\.[a-z]{2})?)\b'

    cleaned_text = re.sub(suffix_pattern, '', text, flags=re.IGNORECASE)

    # print(text,",",cleaned_text)
    return cleaned_text

# remove_urls("abd.cds.cas.def")

def confusion_matrix_plot(y,pred):
  cm = confusion_matrix(y, pred)
  cm[0][0],cm[1][1]=cm[1][1],cm[0][0]
  sns.heatmap(cm, annot=True,xticklabels=["Matching","Not matching"],yticklabels=["Matching","Not matching"],cmap="Reds")
  return cm

def accuracy(y,pred):
  acc=0
  for i in range(len(y)):
    acc+=(y[i]==pred[i])
  return acc/len(y)


def predict_dataset(df):
  data=[]
  for i in range(len(df)):
    s1=df.loc[i,"key1"]
    s2=df.loc[i,"key2"]
    data.append([s1,s2])

  obj=checksimilarity()
  res=obj.mean_ensemble_model(data)
  return res


def evaluate_dataset(df):
  y=[]
  for i in range(len(df)):
    if(df.loc[i,"is_correct_match"]):
      y.append(1)
    else:
      y.append(0)
  res=predict_dataset(df)
  pred=[]
  for i in range(len(res)):
    if(res[i]>=0.5):
      pred.append(1)
    else:
      pred.append(0)

  confusion_matrix_plot(y,pred)
  return accuracy(y,pred)




def cartessian_product(l1,l2):
  mat=[]
  for i in range(len(l1)):
    for j in range(len(l2)):
      mat.append([l1[i],l2[j]])
  return np.array(mat)


def preprocess_data(df):
  id_maps={}
  temp=[]
  y=[]
#   for i in range(len(df)):
#     if(df.loc[i,"is_correct_match"]):
#       y.append(1)
#     else:
#       y.append(0)


  # print(df.loc[18,"key1"])
  df["key1"]=df["key1"].apply(lambda x: remove_urls(x))
  df["key2"]=df["key2"].apply(lambda x: remove_urls(x))

  # df["key1"]=df["key1"].apply(lambda x: " ".join([word for word in x.split() if word not in english_stopwords]))
  # df["key2"]=df["key2"].apply(lambda x: " ".join([word for word in x.split() if word not in english_stopwords]))

  df["key1"]=df["key1"].apply(lambda x: x.replace("@"," "))
  df["key2"]=df["key2"].apply(lambda x: x.replace("@"," "))

  df["key1_upd"]=df["key1"].apply(lambda x: split_sentence(x,2))
  df["key2_upd"]=df["key2"].apply(lambda x: split_sentence(x,2))
  print("*****" ,"done")

    # s1=" ".join([word for word in s1.split() if word not in english_stopwords])
    # s2 = " ".join([word for word in s2.split() if word not in english_stopwords])

  data=np.zeros((0,2))
  s=0
  # print(len(df))
  for i in range(len(df)):
    temp=cartessian_product(df.loc[i,"key1_upd"],df.loc[i,"key2_upd"])
    # print(s,len(df),i)
    for j in range(len(temp)):
      id_maps.update({s+j:i})
    data=np.vstack((data,temp))
    s+=len(temp)


  print(s,len(df))


  newdf=pd.DataFrame(data)
  newdf.columns=["key1","key2"]
  predictions=predict_dataset(newdf)
  pred=[0 for i in range(len(df))]
  return predictions
#   print("Misclassified samples : ")
#   for i in range(len(predictions)):
#     if(predictions[i]>=0.50):
#       # print(newdf.loc[i,"key1"]," , ",newdf.loc[i,"key2"])
#       pred[id_maps[i]]=1


#   for i in range(len(df)):
#     if(pred[i]!=y[i]):
#       print(df.loc[i,"key1_upd"]," , ",df.loc[i,"key2_upd"],i)
#   print(accuracy(y,pred))
#   confusion_matrix_plot(y,pred)



