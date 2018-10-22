from nltk import word_tokenize
from nltk.stem import PorterStemmer
from porter_stemmer import MartinPorterStemmer
from numpy import dot
from numpy.linalg import norm

from functools import reduce
from math import sqrt
import re

def sum_squared(values):
    return reduce(lambda x, y: x + y, map(lambda x: x ** 2, list(values)))

def tokenify(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    # words = [word for word in words if word.isalpha()]
    tokens = stem_martin(words)
    # tokens = stem_nltk(words)

    return tokens

def stem_martin(words):
  stemmer = MartinPorterStemmer()
  return [stemmer.stem(word, 0,len(word)-1) for word in words]


def stem_nltk(words):
  stemmer = PorterStemmer()
  return [stemmer.stem(word) for word in words]

def cos_similiarity(a, b):
  norm_ab = (norm(a)*norm(b))
  if norm_ab == 0:
    return 0
  else:
    return dot(a, b) / norm_ab
