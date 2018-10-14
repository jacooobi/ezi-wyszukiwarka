def read_file(name):
  try:
    with open(name, 'r') as f:
        data = f.read()
    f.close()
    return data
  except:
    print('Error occured while loading data. Closing the search engine...')
    exit();
