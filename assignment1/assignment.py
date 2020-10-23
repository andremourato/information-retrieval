import csv
import re
import Stemmer
# from memory_profiler import profile
import tracemalloc
import time
import sys

def load_stop_words(file):
    with open(file)  as f_in:
        return [ _.split()[0] for _ in f_in ]

def read_corpus(file):
    with open(file)  as csvfile:
        reader = csv.DictReader(csvfile)
        documentList = []
        for row in reader:
            if len(row['abstract']) > 0:
                documentList.append(row['title'] + row['abstract'])

    return documentList

# def remove_chars(string, char_list=['(',')','&ndash;',\
#                                     '.',',','%','[',']',\
#                                     ':','/','\'','\\','-']):
#     return ' '.join([string.replace(c,' ') for c in char_list])

def remove_chars(string):
    # table = str.maketrans({"(":'', ")":'',"{":'',"}":'',"[":'',"]":'',"-":'',"#":'',"%":'',"$":'',"&":'', "/":'', ".":'', ",":''})
    # return string.translate(table)
    return ''.join([ c if c.isalpha() or c.isnumeric() else ' ' for c in string])

def filter_stop_words(lst):
    return [doc for doc in lst if doc not in stopwords]

def simple_tokenizer(lst):
    # replaces all non-alphabetic characters by a space, 
    # lowercases tokens,
    # splits on whitespace, and
    # ignores all tokens with less than 3 characters
    return [[token for token in remove_chars(document).lower().split() if len(token) >= 3] for document in lst]

def improved_tokenizer(lst):
    stemmer = Stemmer.Stemmer('porter')
    return filter_stop_words([ stemmer.stemWords(document.split()) for document in lst ])

# @profile(precision=4)
def indexer(lst):
    count_index = {}
    document_index = {}
    for docID in range(len(lst)):
        for token in lst[docID]:
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
tracemalloc.start()
#1 - Reads the file
lst = read_corpus(filename)
#2 - Loads stop words
stopwords = load_stop_words('stopwords.txt')
#2 - Applies the tokenizer
#2a - Applies the simple tokenizer
if indexer_mode == 1:
    index = simple_tokenizer(lst[:3])
else:#2b - Applies the improved tokenizer
    index = improved_tokenizer(lst[:3])

#3 - Creates an indexing pipeline
time_start = time.process_time()
count_index, document_index = indexer(index)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
print('Elapsed:',time.process_time() - time_start,'s')

print(count_index)
#print(document_index)