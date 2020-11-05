###############################
#   Authors             
###############################
#   André Mourato nmec 84745
#   Gonçalo Marques nmec 80327
###############################
import csv
import tracemalloc
import time
import sys

def load_stop_words(file):
    with open(file)  as f_in:
        return [ _.split()[0] for _ in f_in ]

def remove_non_alpha(string):
    return ''.join([ c if c.isalpha() else ' ' for c in string.lower()]).split() 

def simple_index(filename):
    '''A tokenizer that replaces all non-alphabetic characters by a space, lowercases
    tokens, splits on whitespace, and ignores all tokens with less than 3 characters. 
    Parameters
    ----------
    filename : string
        File containing the dataset
        
    Returns
    -------
    count_index : dict
        Dictionary that contains the token as the key and the number of occurences as the value.
        Example: {
            'incub': 4,
            'period': 6,
            'epidemiolog': 2,
            'characterist': 2,
            'novel': 5,
            'coronaviru': 8,
        }
    document_index : dict
        Dictionary that contains the token as the key and the list of documents as the value.
        Example: {
            'incub': {8: true, 2: True},
            'period': {8: True, 2: True, 5: True}
        }
    '''
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
                    # and the document_index dict which registers the list of documents
                    # in which a token appears
                    count_index[tok] = count_index.setdefault(tok, 0) + 1
                    # Created dictionary with each value being {idx: True}, in order
                    # to make sure that a token only refers to a document once, even
                    # though it may appear more than once in it. We used a dictionary
                    # instead of a list in order to improve performance
                    if tok not in document_index:
                        document_index[tok] = {}
                    document_index[tok][idx] = True
    
    return count_index, document_index

def improved_index(filename):
    '''An improved tokenizer that replaces all non-alphabetic characters by a space, lowercases
    tokens, splits on whitespace, and ignores all tokens with less than 3 characters. This tokenizer
    also uses the Porter stemmer and applies a stopword filter
    ----------
    filename : string
        File containing the dataset
        
    Returns
    -------
    count_index : dict
        Dictionary that contains the token as the key and the number of occurences as the value.
        Example: {
            'incub': 4,
            'period': 6,
            'epidemiolog': 2,
            'characterist': 2,
            'novel': 5,
            'coronaviru': 8,
        }
    document_index : dict
        Dictionary that contains the token as the key and the list of documents as the value.
        Example: {
            'incub': {8: true, 2: True},
            'period': {8: True, 2: True, 5: True}
        }
    '''
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
                # tokens, splits on whitespace, and ignores all tokens with less than 3 characters.
                # This tokenizer also uses the Porter stemmer and applies a stopword filter
                for tok in Stemmer.Stemmer('porter').stemWords([token \
                    for token in (remove_non_alpha(string)) \
                        if len(token) >= 3 and token not in stopwords]):
                    
                    # Indexes all the input tokens into two dictionaries
                    # the count_index dict which registers the total number of occurrences
                    # of a token and the document_index dict which registers the list of
                    # documents in which a token appears
                    count_index[tok] = count_index.setdefault(tok, 0) + 1
                    # Created dictionary with each value being {idx: True}, in order
                    # to make sure that a token only refers to a document once, even
                    # though it may appear more than once in it. We used a dictionary
                    # instead of a list in order to improve performance
                    if tok not in document_index:
                        document_index[tok] = {}
                    document_index[tok][idx] = True
    
    return count_index, document_index

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'all_sources_metadata_2020-03-13.csv'
    else:
        filename = sys.argv[1]
    print('Reading from file',filename)

    print('-----------------------------')
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

    print('\n------------------------------')
    print('IMPROVED TOKENIZER')
    count_index = {}
    document_index = {}

    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
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