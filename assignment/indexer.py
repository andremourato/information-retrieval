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
        # 1 - NON-NORMALIZED WEIGHT CALCULATION
        document_term_weights[docID] = {}
        for token in term_index[docID]:
            document_term_weights[docID][token] = 1 + math.log10(term_index[docID][token])
            if token not in term_document_weights:
                term_document_weights[token] = {}
            term_document_weights[token][docID] = 1 + math.log10(term_index[docID][token])
    
        # 2 - CALCULATION OF THE NORM FACTOR
        norm_factor = 1/math.sqrt(sum([w**2 for w in document_term_weights[docID].values()]))
        
        # 3 - NORMALIZED WEIGHT CALCULATION
        for token in document_term_weights[docID]:
            document_term_weights[docID][token] *= norm_factor
            term_document_weights[token][docID] *= norm_factor
            # 4 - Calculating IDF
            N = len(document_length_index)
            dft = len(term_document_weights[token])
            idf_list[token] = math.log10(N/dft)
    return document_term_weights, term_document_weights, idf_list

def bm25_avdl(document_length_index):
    avdl = sum(v for v in document_length_index.values())
    return avdl / len(document_length_index)

def bm25_weighting(N, k, b, avdl, document_term_index, document_length_index, idf_list):
    weights = {}
    for docID in document_term_index:
        for token in document_term_index[docID]:
            if token not in weights:
                weights[token] = {}
            # Calculates the weight of term token in docID
            first = idf_list[token]
            second = (k+1) * document_term_index[docID][token]
            third = 1 / ( k*((1-b) + (b*document_length_index[docID] / avdl)) + document_term_index[docID][token])
            weights[token][docID] = first*second*third 
    return weights
                

def bmc_pre_calculation(term_index, document_length_index, idf_list):
    avdl = bm25_avdl(document_length_index)
    N = len(document_length_index)
    k = 1.2
    b = 0.75
    return bm25_weighting(N, k, b, avdl, term_index, document_length_index, idf_list)

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
    
    # 2 - LNC 
    document_term_weights, term_document_weights, idf_list = lnc_calculation(term_index,document_length_index)
    dump_weights(term_document_weights, idf_list, 'tf_idf_weights.csv')

    # 3 - BMC
    bmc_weights = bmc_pre_calculation(term_index,document_length_index, idf_list)
    dump_weights(bmc_weights, idf_list, 'bmc_weights.csv')

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
    dump_to_file(term_index,'term_index.json')

    dump_to_file(document_term_weights,'document_term_weights.json')

    dump_to_file(term_document_weights,'term_document_weights.json')
    
    dump_to_file(document_length_index,'document_length_index.json')

    dump_to_file(idf_list,'idf_list.json')

    