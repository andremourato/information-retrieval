##########################
#   Authors             
##########################
#   André Mourato
#   Gonçalo Marques
##########################
import csv
import tracemalloc
import time
import sys

def load_stop_words(file):
    with open(file)  as f_in:
        return [ _.split()[0] for _ in f_in ]

def simple_index(filename):
    count_index = {}
    document_index = {}
    with open(filename) as csvfile:

        # Iterate over the CSV file ignoring entries without an abstract
        # and joining the title and abstract fields into a single string
        for idx,row in enumerate(csv.DictReader(csvfile)):
            if len(row['abstract']) > 0:
                string =  row['title'] + ' ' + row['abstract'] 

                # Removes non-alphabetic characters, sets string to lower case and
                # removes word with less than 3 characters
                for tok in [token for token in (remove_non_alpha(string)) if len(token) >= 3]:
                   
                    # Indexes all the input tokens into two dictionaries
                    # the count_index dict which registers the total number of occurrences of a token
                    # and the document_index dict which registers the list of documents in which a token appears
                    count_index[tok] = count_index.setdefault(tok, 0) + 1
                    document_index.setdefault(tok,[]).append(idx)
    
    return count_index, document_index

def improved_index(filename):
    count_index = {}
    document_index = {}
    stopwords = load_stop_words('stopwords.txt')
    import Stemmer
    with open(filename) as csvfile:
    
        # Iterate over the CSV file ignoring entries without an abstract
        # and joining the title and abstract fields into a single string
        for idx,row in enumerate(csv.DictReader(csvfile)):
            if len(row['abstract']) > 0:
                string =  row['title'] + ' ' + row['abstract'] 

                # Removes non-alphabetic characters by a space, lowercases
                # tokens, splits on whitespace, and ignores all tokens with less than 3 characters. This tokenizer
                # also uses the Porter stemmer and applies a stopword filter
                for tok in Stemmer.Stemmer('porter').stemWords([token for token in (remove_non_alpha(string)) if len(token) >= 3 and token not in stopwords]):
                    
                    # Indexes all the input tokens into two dictionaries
                    # the count_index dict which registers the total number of occurrences of a token
                    # and the document_index dict which registers the list of documents in which a token appears
                    count_index[tok] = count_index.setdefault(tok, 0) + 1
                    document_index.setdefault(tok,[]).append(idx)
    
    return count_index, document_index

def remove_non_alpha(string):
    return ''.join([ c if c.isalpha() else ' ' for c in string.lower()]).split() 


if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'all_sources_metadata_2020-03-13.csv'
    else:
        filename = sys.argv[1]
    print('Reading from file',filename)

    print('SIMPLE TOKENIZER')
    count_index = {}
    document_index = {}

    # Trace used memory and time for the 
    # indexing process using the simple tokenizer
    tracemalloc.start()
    time_start = time.process_time()
    
    count_index, document_index = simple_index(filename)
    
    print('a) Total indexing time:',time.process_time() - time_start,'s')
    current, peak = tracemalloc.get_traced_memory()
    print(f"a) Memory usage for indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()

    print('\nb) Total vocabulary size is: ',len(count_index),'words')
    print('\nc) First 10 alphabetically ordered terms with document frequency = 1:')
    print(sorted([key for key in document_index if len(document_index[key]) == 1])[:10])
    print('\nd) 10 terms with the highest document frequency:')
    print(sorted(document_index, key = lambda key: len(document_index[key]))[-10:])

    print('\n\n------------------------------\n\n')
    print('IMPROVED TOKENIZER')
    count_index = {}
    document_index = {}

    # Trace used memory and time for the 
    # indexing process using the simple tokenizer
    tracemalloc.start()
    time_start = time.process_time()

    count_index, document_index = improved_index(filename)
    
    print('a) Total indexing time:',time.process_time() - time_start,'s')
    current, peak = tracemalloc.get_traced_memory()
    print(f"a) Memory usage for indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()

    print('\nb) Total vocabulary size is: ',len(count_index),'words')
    print('\nc) First 10 alphabetically ordered terms with document frequency = 1:')
    print(sorted([key for key in document_index if len(document_index[key]) == 1])[:10])
    print('\nd) 10 terms with the highest document frequency:')
    print(sorted(document_index, key = lambda key: len(document_index[key]))[-10:])