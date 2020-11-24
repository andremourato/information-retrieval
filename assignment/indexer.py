###############################
#   Authors             
###############################
#   André Mourato nmec 84745
#   Gonçalo Marques nmec 80327
###############################
# Benchmarking
import tracemalloc
import time
# Necessary imports
import sys
import math
import Stemmer
import operator
import csv
# File imports
from utils import *

def indexer(filename):
    '''An improved tokenizer that replaces all non-alphabetic characters by a space, lowercases
    tokens, splits on whitespace, and ignores all tokens with less than 3 characters. This tokenizer
    also uses the Porter stemmer and applies a stopword filter
    ----------
    filename : string
        File containing the dataset
        
    Returns
    -------
    term_index : dict
        Dictionary that contains the token as the key and the number of occurences as the value.
        Example: {
            "9dj07sac": {
                "incub": 4,
                "period": 4,
                "epidemiolog": 2,
                "characterist": 1,
                "novel": 2
            }
        }
    '''
    term_index = {}
    document_length_index = {}
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
                    
                    # Indexes all the input tokens into one dictionaries
                    # the term_index dict which registers the total number of occurrences
                    # of a token in each document
                    # Counts the number of tokens in each document
                    if row['cord_uid'] not in document_length_index:
                        document_length_index[row['cord_uid']] = 0
                    # Counts the number of terms in each document
                    document_length_index[row['cord_uid']] += 1
                    # Counts the term frequency
                    if row['cord_uid'] not in term_index:
                        term_index[row['cord_uid']] = {}
                    if tok not in term_index[row['cord_uid']]:
                        term_index[row['cord_uid']][tok] = 1
                    else:
                        term_index[row['cord_uid']][tok] += 1
    
    return term_index, document_length_index

def lnc_calculation(term_index,document_length_index):
    document_term_weights = {}
    term_document_weights = {}
    idf_list = {}
    for docID in term_index:
        for token in term_index[docID]:
            if docID not in document_term_weights:
                document_term_weights[docID] = {}
            document_term_weights[docID][token] = 1 + math.log10(term_index[docID][token])
            if token not in term_document_weights:
                term_document_weights[token] = {}
            term_document_weights[token][docID] = 1 + math.log10(term_index[docID][token])
    
        norm_factor = 1/math.sqrt(sum([document_term_weights[docID][_]**2 for _ in document_term_weights[docID]]))
        for token in document_term_weights[docID]:
            document_term_weights[docID][token] *= norm_factor
            term_document_weights[token][docID] *= norm_factor
            # Calculating idf of each term
            N = len(document_length_index)
            dft = len(term_document_weights[token])
            idf_list[token] = math.log10(N/dft)
    return document_term_weights, term_document_weights, idf_list

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'datasets/metadata_2020-03-27.csv'
    else:
        filename = sys.argv[1]
    print('Reading dataset from file',filename)
    print('------------------------------------------------------------')
    print('STARTING INDEXING...')
    print('------------------------------------------------------------')

    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
    tracemalloc.start()
    time_start = time.process_time()

    #########################################################
    # LOADING INFORMATION FROM A FILE
    #########################################################
    stopwords = load_stop_words('resources/stopwords.txt')

    #########################################################
    # INDEXER
    #########################################################
    # 1 - Indexing
    term_index, document_length_index = indexer(filename)
    
    # 2 - LNC calculation
    document_term_weights, term_document_weights, idf_list = lnc_calculation(term_index,document_length_index)

    # 3 - Dumping to file
    dump_term_idf_weights(term_document_weights, idf_list)
    
    #########################################################
    # BENCHMARKING INFORMATION
    #########################################################
    print('Total indexing time:',time.process_time() - time_start,'s')
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Memory usage for indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    print('Total vocabulary size is: ',len(term_index),'words')
    print('------------------------------------------------------------')

    #########################################################
    # DUMPING DATA STRUCTURES TO A FILE
    #########################################################
    # dump_to_file(term_index,'term_index.json')

    # dump_to_file(document_term_weights,'document_term_weights.json')

    dump_to_file(term_document_weights,'term_document_weights.json')
    
    # dump_to_file(document_length_index,'document_length_index.json')

    dump_to_file(idf_list,'idf_list.json')
        
    # dump_to_file(query_weights,'query_weights.json')

    # dump_to_file(document_query_weights,'document_query_weights.json')

    # dump_to_file(scores,'scores.json')
    