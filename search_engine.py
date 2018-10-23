import argparse
from helpers import sum_squared, tokenify, cos_similiarity
from utils import read_file
from math import log10, sqrt, log
from operator import itemgetter


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("-d", "--documents", required=True, help="Path to the documents file")
  ap.add_argument("-k", "--keywords", required=True, help="Path to the keywords file")
  ap.add_argument("-p", "--print_processed", default=False, help="Print processed docs and terms")
  ap.add_argument("-s", "--stemmer", required=False, help="Type of stemmer to use")

  return vars(ap.parse_args())

def setup_keywords(data):
  keywords_vector = {}

  for keyword in tokenify(data):
    keywords_vector[keyword]= 0

  return keywords_vector

def setup_documents(data):
  docs = data.split('\n\n')
  documents = []
  body = ''

  for i, doc in enumerate(docs):
    doc_data = doc.split('\n')
    title = doc_data[0]

    if len(doc_data) > 1:
      body = ' '.join(doc_data[1:len(doc_data)])

    documents.append({'idx': i, 'title': title, 'body': body, 'tokens': tokenify(doc)})

  return documents

def setup_database(documents_path, keywords_path):
  documents = setup_documents(read_file(documents_path))
  keywords = setup_keywords(read_file(keywords_path))

  return documents, keywords

def calculate_bag_of_words(tokens, keywords):
  keywords_bag = keywords.copy()

  for token in tokens:
    if token in keywords.keys():
      keywords_bag[token] += 1

  return keywords_bag

def calculate_tf(bag, keywords):
  max_n = max(bag.values())

  tf_vector = keywords.copy()

  for key in tf_vector.keys():
    try:
      tf_vector[key] = bag[key] / max_n
    except ZeroDivisionError:
      tf_vector[key] = 0.0

  tf_vector_length = sqrt(sum_squared(tf_vector.values()))

  return tf_vector, tf_vector_length

def calculate_idf(documents, keywords):
  idf = keywords.copy()
  document_count = len(documents)

  for key in idf.keys():
    word_count = 0

    for doc in documents:
      if key in set(doc['tokens']):
        word_count += 1

    try:
      idf[key] = log10(document_count / word_count)

    except (ValueError, ZeroDivisionError) as e:
      idf[key] = 0

  return idf

def calculate_query_tf_idf(tokens, keywords, idf):
  tfidf_vector = keywords.copy()

  for key in tfidf_vector.keys():
    tfidf_vector[key] = tokens[key] * idf[key]

  tfidf_vector_length = sqrt(sum_squared(tfidf_vector.values()))

  return tfidf_vector, tfidf_vector_length

def setup_documents_bag_matrix(documents, keywords):
  bag_matrix = []

  for i, doc in enumerate(documents):
    bag = calculate_bag_of_words(doc['tokens'], keywords)
    bag_matrix.append({'idx': i, 'vector': bag})

  return bag_matrix

def setup_documents_tf_matrix(bag_matrix, keywords):
  tf_matrix = []

  for i, bag in enumerate(bag_matrix):
    tf, tf_length = calculate_tf(bag['vector'], keywords)
    tf_matrix.append({'idx': i, 'vector': tf, 'vector_length': tf_length})

  return tf_matrix

def setup_documents_tfidf_matrix(tf_matrix, keywords, idf):
  tfidf_matrix = []

  for i, document in enumerate(tf_matrix):
    tfidf_vector = keywords.copy()

    for key in document['vector'].keys():
      tfidf_vector[key] = document['vector'][key] * idf[key]

    tfidf_vector_length = sqrt(sum_squared(tfidf_vector.values()))

    tfidf_matrix.append({'idx': i, 'vector': tfidf_vector, 'vector_length': tfidf_vector_length})

  return tfidf_matrix

def parse_query(input_text):
  return tokenify(input_text)

def print_processed_docs_and_terms(documents, keywords):
  print("Processed terms: \n")

  for keyword in sorted(keywords.keys()):
    print(keyword, end=' ')

  print('\n\n')

  for doc in documents:

    print('Title: ', doc['title'], '\n')

    for token in sorted(set(doc['tokens'])):
      print(token, end=' ')

    print('\n')

def calculate_scores(documents_tfidf, query_tfidf, query_tfidf_len):
  scores = []
  for doc_tfidf in documents_tfidf:
    sim = cos_similiarity(list(doc_tfidf['vector'].values()), list(query_tfidf.values()))

    scores.append({'idx': doc_tfidf['idx'], 'score': sim})

  return sorted(scores, key=itemgetter('score'), reverse=True)

def print_result(scores, documents):
  for score in scores:
    print("{} {:.8f}".format(documents[score['idx']]['title'].ljust(100), score['score']))

def print_candidate_set(scores, documents, size):
  for i, score in enumerate(scores):
    if i >= size:
      return
    print("{}: {}".format(scores[i]['idx'], documents[score['idx']]['title'].ljust(100)))


def get_top_ids(scores, n = 5):
  top_ids = []

  for i, doc in enumerate(scores):
    if i >= n:
      return top_ids
    top_ids.append(str(doc['idx']))

def split_user_feedback(user_ranking, documents, scores, keywords):
  relevant_vector = keywords.copy()
  non_relevant_vector = keywords.copy()

  relevant = user_ranking.split(' ')
  non_relevant = list(set(get_top_ids(scores, 5)).difference(set(relevant)))

  for i in relevant:
    for word in documents[int(i)]['tokens']:
      if word in keywords.keys():
        relevant_vector[word] += 1

  for i in non_relevant:
    for word in documents[int(i)]['tokens']:
      if word in keywords.keys():
        non_relevant_vector[word] += 1

  return relevant_vector, non_relevant_vector

def calculate_new_query(initial_query, relevant_keys, non_relevant_keys):
  beta = 0.75
  alpha = 1
  gamma = 0.25

  final_query = initial_query.copy()

  for word in final_query.keys():
    score = alpha*initial_query[word] + beta*relevant_keys[word] - gamma*non_relevant_keys[word]
    if score < 0:
      final_query[word] = 0
    else:
      final_query[word] = score

  return final_query

def print_updated_query(final_query):
  for key in final_query.keys():
    if final_query[key] > 0:
      print("{}: {}".format(key, final_query[key]))

def main(args):
  documents, keywords = setup_database(args['documents'], args['keywords'])

  bag_matrix = setup_documents_bag_matrix(documents, keywords)
  tf_matrix = setup_documents_tf_matrix(bag_matrix, keywords)
  idf = calculate_idf(documents, keywords)
  documents_tfidf = setup_documents_tfidf_matrix(tf_matrix, keywords, idf)

  if args['print_processed']:
    print_processed_docs_and_terms(documents, keywords)
  else:
    while True:
      input_text = input("Please insert query here (type exit() to quit) ")

      if input_text == 'exit()':
        print("Closing the search engine...")
        exit()

      query_tokens = parse_query(input_text)
      query_bag_vector = calculate_bag_of_words(query_tokens, keywords)

      query_tf, query_tf_length = calculate_tf(query_bag_vector, keywords)
      query_tfidf, query_tfidf_length = calculate_query_tf_idf(query_tf, keywords, idf)

      scores = calculate_scores(documents_tfidf, query_tfidf, query_tfidf_length)

      print_candidate_set(scores, documents, 5);

      user_ranking = input("Which of the documents are relevant? Use indexes listed above (e.g. 5 10 15) ")

      relevant_vector, non_relevant_vector = split_user_feedback(user_ranking, documents, scores, keywords)

      final_query = calculate_new_query(query_bag_vector, relevant_vector, non_relevant_vector)

      query_tf, query_tf_length = calculate_tf(final_query, keywords)
      query_tfidf, query_tfidf_length = calculate_query_tf_idf(query_tf, keywords, idf)

      scores = calculate_scores(documents_tfidf, query_tfidf, query_tfidf_length)

      print("Congratulations. Your Search results are presented below!\n")
      print_result(scores, documents)
      print_updated_query(final_query)


if __name__ == '__main__':
  main(parse_args())
