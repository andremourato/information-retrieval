##########################
#   Authors             
##########################
#   André Mourato
#   Gonçalo Marques
##########################
import csv
import Stemmer
import tracemalloc
import time
import sys

def load_stop_words(file):
    with open(file)  as f_in:
        return [ _.split()[0] for _ in f_in ]

def read_corpus(file):
    with open(file) as csvfile:
        return {
            row['doi'] : row['title'] + ' ' + row['abstract'] \
                for idx,row in enumerate(csv.DictReader(csvfile)) \
                    if len(row['abstract']) > 0 and idx < 3
        }

def remove_chars(string):
    return ''.join([ c if c.isalpha() else ' ' for c in string])

def filter_stop_words(lst):
    return [doc for doc in lst if doc not in stopwords and len(doc) >= 3]

def simple_tokenizer(document_dict):
    # replaces all non-alphabetic characters by a space, 
    # lowercases tokens,
    # splits on whitespace, and
    # ignores all tokens with less than 3 characters
    # return [[token for token in remove_chars(document).lower().split() if len(token) >= 3] for document in lst]
    return {
        docID : [token for token in remove_chars(string).lower().split() if len(token) >= 3] \
            for (docID,string) in document_dict.items()
    }

def improved_tokenizer(document_dict):
    stemmer = Stemmer.Stemmer('porter')
    return {
        docID: stemmer.stemWords(filter_stop_words(remove_chars(string).lower().split())) \
            for (docID,string) in document_dict.items()
    }

def indexer(document_dict):
    count_index = {}
    document_index = {}
    for docID in document_dict:
        for token in document_dict[docID]:
            if token not in count_index.keys():
                count_index[token] = 1
            else:
                count_index[token] += 1
            if token not in document_index.keys():
                document_index[token] = {}
            document_index[token][docID] = True
    return count_index, document_index

indexer_mode = 0
if len(sys.argv) >= 2:
    print('RUNNING THE SIMPLE TOKENIZER...')
    indexer_mode = 1
else:
    print('RUNNING THE IMPROVED TOKENIZER...')

filename = 'all_sources_metadata_2020-03-13.csv'

#1 - Reads the file
documents = read_corpus(filename)

#2 - Loads stop words
stopwords = load_stop_words('stopwords.txt')
#2 - Applies the tokenizer
#2a - Applies the simple tokenizer

if indexer_mode == 1:
    documents = simple_tokenizer(documents)
else:#2b - Applies the improved tokenizer
    documents = improved_tokenizer(documents)

#3 / 4.a - Creates an indexing pipeline and monitors how much time and memory were used in the indexing process
tracemalloc.start()
time_start = time.process_time()
count_index, document_index = indexer(documents)
current, peak = tracemalloc.get_traced_memory()
print('a) Total indexing time:',time.process_time() - time_start,'s')
print(f"a) Memory usage for indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
#4.b - Vocabulary size
print('\nb) Total vocabulary size is: ',len(count_index),' words')

#4.c - 10 terms with document frequency = 1 alphabetically ordered
print('\nc) First 10 alphabetically ordered terms with document frequency = 1:')
print(sorted([key for key in document_index if len(document_index[key]) == 1])[:10])

#4.d - 10 terms with the highest document frequency
print('\nd) 10 terms with the highest document frequency:')
print(sorted(document_index, key = lambda key: len(document_index[key]))[-10:])