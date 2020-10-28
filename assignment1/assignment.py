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

def read_corpus(filename):
    '''Loads the dataset from a file
    Parameters
    ----------
    filename : string
        Name of the file that contains the dataset.
        
    Returns
    -------
    dict : dict
        Dictionary that associates the doi of documents to their title + abstract.

    The output has the following format:
    {
        'doi-1111': 'Document 1 This is the abstract',
        'doi-2222': 'Document 2 This is the abstract',
        'doi-3333': 'Document 3 This is the abstract'
    }
    '''
    with open(filename) as csvfile:
        dic = {}
        for idx,row in enumerate(csv.DictReader(csvfile)):
            if len(row['abstract']) > 0:
                dic[idx] =  row['title'] + ' ' + row['abstract']

        return dic

def get_filtered_tokens(string):
    '''Removes non-alphabetic characters, sets string to lower case and
    removes word with less than 3 characters
    Parameters
    ----------
    string : string
        The string from where the tokens will be extracted
        
    Returns
    -------
    tokens : list
        The list of tokens that were extracted from the input string

    get_filtered_tokens('within a larger purine run (A(n)G(n)) in this process, by a mechanism')
    would return ['within', 'larger', 'purine', 'run', 'this', 'process', 'mechanism']
    '''
    return [token for token in ''.join([ c if c.isalpha() else ' ' for c in string.lower()]).split()\
                if len(token) >= 3]

def filter_stop_words(tokens):
    '''Removes the stop words from the list of input tokens
    Parameters
    ----------
    tokens : list
        The list of tokens
        
    Returns
    -------
    tokens_without_stopwords : list
        The list of tokens without the stop words

    filter_stop_words(['within', 'larger', 'purine', 'run', 'this', 'process', 'mechanism'])
    would return ['within', 'larger', 'purine', 'run', 'process', 'mechanism']
    '''
    return [token for token in tokens if token not in stopwords]

def simple_tokenizer(document_dict):
    '''A tokenizer that replaces all non-alphabetic characters by a space, lowercases
    tokens, splits on whitespace, and ignores all tokens with less than 3 characters. 
    Parameters
    ----------
    document_dict : dict
        Dictionary that associates the doi of documents to their title + abstract.
        
    Returns
    -------
    document_dict : dict
        Dictionary that associates the doi of documents to the list of tokens
        contained in it
    '''
    return {
        docID : get_filtered_tokens(string) \
            for (docID,string) in document_dict.items()
    }

def improved_tokenizer(document_dict):
    '''An improved tokenizer that replaces all non-alphabetic characters by a space, lowercases
    tokens, splits on whitespace, and ignores all tokens with less than 3 characters. This tokenizer
    also uses the Porter stemmer and applies a stopword filter
    Parameters
    ----------
    document_dict : dict
        Dictionary that associates the doi of documents to their title + abstract.
        
    Returns
    -------
    document_dict : dict
        Dictionary that associates the doi of documents to the list of tokens
        contained in it
    '''
    return {
        docID: Stemmer.Stemmer('porter').stemWords(filter_stop_words(get_filtered_tokens(string))) \
            for (docID,string) in document_dict.items()
    }

def indexer(document_dict):
    '''Indexes all the input tokens, registering the total number of occurrences of a token
    and the list of documents in which it appears
    Parameters
    ----------
    document_dict : dict
        Dictionary that associates the doi of documents to the list of tokens
        
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
            'incub': ['10.3390/jcm9020538', '10.3390/jcm9020538', '10.3390/jcm9020538'],
            'period': ['10.3390/jcm9020538']
        }
    '''
    count_index = {}
    document_index = {}
    for docID in document_dict:
        for token in document_dict[docID]:
            if token not in count_index:
                count_index[token] = 1
            else:
                count_index[token] += 1
            if token not in document_index:
                document_index[token] = [docID]
            else:
                document_index[token].append(docID)
    return count_index, document_index

if __name__ == '__main__':

    if len(sys.argv) < 2:
        filename = 'all_sources_metadata_2020-03-13.csv'
    else:
        filename = sys.argv[1]
    print('Reading from file',filename)

    #0 - Reads the file
    documents = read_corpus(filename)

    #1 - Loads stop words
    stopwords = load_stop_words('stopwords.txt')

    #2 - Applies the tokenizer
    for indexer_mode in range(2):
        #2.i - Applies the simple tokenizer
        if indexer_mode == 0:
            print('--------------------- RUNNING THE SIMPLE TOKENIZER ---------------------')
            tokens = simple_tokenizer(documents)
        else:#2.ii - Applies the improved tokenizer
            print('\n\n--------------------- RUNNING THE IMPROVED TOKENIZER ---------------------')
            tokens = improved_tokenizer(documents)

        #3 / 4.a - Creates an indexing pipeline and monitors how much time and memory were used in the indexing process
        tracemalloc.start()
        time_start = time.process_time()
        count_index, document_index = indexer(tokens)
        print('a) Total indexing time:',time.process_time() - time_start,'s')
        current, peak = tracemalloc.get_traced_memory()
        print(f"a) Memory usage for indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()
        #4.b - Vocabulary size
        print('\nb) Total vocabulary size is: ',len(count_index),'words')

        #4.c - 10 terms with document frequency = 1 alphabetically ordered
        print('\nc) First 10 alphabetically ordered terms with document frequency = 1:')
        print(sorted([key for key in document_index if len(document_index[key]) == 1])[:10])

        #4.d - 10 terms with the highest document frequency
        print('\nd) 10 terms with the highest document frequency:')
        print(sorted(document_index, key = lambda key: len(document_index[key]))[-10:])
