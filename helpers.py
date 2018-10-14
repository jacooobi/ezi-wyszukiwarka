from nltk import word_tokenize
from nltk.stem import PorterStemmer
from functools import reduce
from math import sqrt

def sum_squared(values):
    return reduce(lambda x, y: x + y, map(lambda x: x ** 2, list(values)))

def tokenify(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    tokens = [stemmer.stem(word) for word in words]

    return tokens

def cos_similiarity(a, b, a_length, b_length):
  ab_sum = 0

  for key in a.keys():
    ab_sum += a[key]*b[key]

  try:
    return ab_sum / a_length * b_length
  except ZeroDivisionError:
    return 0.0
