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

    document_length_index : dict
        Dictionary that contains the number of terms of each document, i.e., the document length
        Example: {
            "vho70jcx": 124,
            "i9tbix2v": 92,
            "62gfisc6": 147,
            "058r9486": 153,
            "wich35l7": 161,
            "z3tgnzth": 88,
            "1xxrnpg3": 127,
        }
    '''
    term_index = {}
    term_index2 = {}
    document_length_index = {}
    block_number = 0
    block_size_limit = 1000000
    with open(filename) as csvfile:
        # Iterate over the CSV file ignoring entries without an abstract
        # and joining the title and abstract fields into a single string

        # documents_count = sum(1 for line in csvfile)
        # csvfile.seek(0)
        
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

                    # # Counts the term frequency
                    # if row['cord_uid'] not in term_index:
                    #     term_index[row['cord_uid']] = {}
                    # if tok not in term_index[row['cord_uid']]:
                    #     term_index[row['cord_uid']][tok] = 1
                    # else:
                    #     term_index[row['cord_uid']][tok] += 1

                    # Counts the term frequency
                    if tok not in term_index2:
                        term_index2[tok] = {}
                    if row['cord_uid'] not in term_index2[tok]:
                        term_index2[tok][row['cord_uid']] = 1
                    else:
                        term_index2[tok][row['cord_uid']] += 1

            if sys.getsizeof(term_index2) > block_size_limit: # or (idx == documents_count-1):
                # temp_dict = sort_terms(term_index2)
                dump_to_file(term_index2,'block_'+ str(block_number) +'.json')
                # temp_dict = {}
                block_number += 1
                term_index2.clear()

        dump_to_file(sorted(term_index2),'block_'+ str(block_number) +'.json')
        
        print('NUMBER OF DOCS ' + str(idx))
    return term_index, document_length_index

# def sort_terms(term_index):
#     # Sorts dictionary terms in alphabetical order
#     print(" -- Sorting terms...")
#     sorted_dictionary = OrderedDict() 
#     return sorted_dictionary

# def merge_blocks():

def lnc_calculation(term_index,document_length_index):
    '''Normalized lnc weight and idf calculator for all terms in dataset
    ----------
    term_index : dict
        Dictionary that contains the token as the key and the number of occurences as the value

    document_length_index : dict
        Dictionary that contains the number of terms of each document, i.e., the document length
        
    Returns
    -------
    term_document_weights : dict
        Dictionary of dictionaries that associates the lnc normalized weight of all
        documents that contain each term
        Example: {
            "siann": {
                "vho70jcx": 0.1646582242160933
            },
            "strain": {
                "vho70jcx": 0.1431574201623654,
                "nljskxut": 0.1081438046556659,
                "lcpp5fim": 0.17536757263696706,
                "gjxumrmm": 0.1248067511230974
                }
            }

    idf_list : dict
        Dictionary that contains the token as the key and the idf as the value.
        Example: {
            "siann": 4.578776695691345,
            "strain": 0.9240227624384145,
            "identif": 1.2125405719730515,
            "align": 2.012928877017827
            }
    '''
    term_document_weights = {}
    idf_list = {}
    for docID in term_index:
        # 1 - NON-NORMALIZED WEIGHT CALCULATION
        document_term_weights = {}
        for token in term_index[docID]:
            document_term_weights[token] = 1 + math.log10(term_index[docID][token])
            if token not in term_document_weights:
                term_document_weights[token] = {}
            term_document_weights[token][docID] = 1 + math.log10(term_index[docID][token])
    
        # 2 - CALCULATION OF THE NORM FACTOR
        norm_factor = 1/math.sqrt(sum([w**2 for w in document_term_weights.values()]))
        
        # 3 - NORMALIZED WEIGHT CALCULATION
        for token in document_term_weights:
            term_document_weights[token][docID] *= norm_factor
            # 4 - Calculating IDF
            N = len(document_length_index)
            dft = len(term_document_weights[token])
            idf_list[token] = math.log10(N/dft)
        
    return term_document_weights, idf_list

def bm25_avdl(document_length_index):
    '''Calculates average document length of the dataset, used for bm25
    ----------
    document_length_index : dict
        Dictionary that contains the token as the key and the number of terms as the value.
        
    Returns
    -------
    avdl : float
        Average document length
    '''
    avdl = sum(v for v in document_length_index.values())
    return avdl / len(document_length_index)

def bm25_weighting(N, k, b, avdl, document_term_index, document_length_index, idf_list):
    '''Calculates bm25 weights for each token
    ----------
    N : int
        Total number of documents

    k : double
        Term frequency saturation value

    b : double
        Document length normalization factor

    avdl : float
        Average document length

    document_term_index : dict
        Dictionary that contains the token as the key and the number of occurences as the value.

    document_length_index : dict
        Dictionary that contains the token as the key and the number of terms as the value.

    idf_list : dict
        Dictionary that contains the token as the key and the idf as the value.
        
    Returns
    -------
        weights : dict
            Dictionary of dictionaries that contains the term as the key 
            and a dictionary with the docIDs in which the term exists and corresponding lnc normalized weight,
            as the value.
        Example :{
            "siann": {
                "vho70jcx": 8.205306314501156
            },
            "strain": {
                "vho70jcx": 1.4736887259523785,
                "nljskxut": 1.050309414805419,
                "lcpp5fim": 1.4952129090104211,
                "gjxumrmm": 1.2983115687733342
                }
            }
    '''
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
    '''Uses parameter values required for bm25 weighting and then calculates bm25 weights for each token
    ----------
    term_index : dict
        Dictionary that contains the token as the key and the number of occurences as the value.

    document_length_index : dict
        Dictionary that contains the token as the key and the number of terms as the value.

    idf_list : dict
        Dictionary that contains the token as the key and the idf as the value.
        
    Returns
    -------
        weights : dict
            Dictionary of dictionaries that contains the term as the key 
            and a dictionary with the docIDs in which the term exists and the corresponding lnc normalized weight,
            as the value.
        Example :{
            "siann": {
                "vho70jcx": 8.205306314501156
            },
            "strain": {
                "vho70jcx": 1.4736887259523785,
                "nljskxut": 1.050309414805419,
                "lcpp5fim": 1.4952129090104211,
                "gjxumrmm": 1.2983115687733342
                }
            }
    '''
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
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage when loading stopwords was {current / 10**6}MB; Peak was {peak / 10**6}MB")

    #########################################################
    # INDEXER
    #########################################################
    # 1 - Indexing
    term_index, document_length_index = indexer(filename)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage when indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    
    # 2 - TF-IDF
    term_document_weights, idf_list = lnc_calculation(term_index,document_length_index)
    dump_weights(term_document_weights, idf_list, 'tf_idf_weights.csv')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage when calculating lnc was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    
    # 3 - BMC
    bmc_weights = bmc_pre_calculation(term_index,document_length_index, idf_list)
    dump_weights(bmc_weights, idf_list, 'bmc_weights.csv')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage when calculating bmc was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    #########################################################
    # BENCHMARKING INFORMATION
    #########################################################
    print('Total indexing time:',time.process_time() - time_start,'s')
    current, peak = tracemalloc.get_traced_memory()
    print(f"FINAL MEMORY USAGE: {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print('Total vocabulary size is: ',len(term_index),'words')
    print('------------------------------------------------------------')

    #########################################################
    # DUMPING DATA STRUCTURES TO A FILE
    # Used for debugging but was kept commented on purpose
    #########################################################
    # dump_to_file(term_index,'term_index.json')

    # dump_to_file(bmc_weights,'bmc_weights.json')
    
    # dump_to_file(document_length_index,'document_length_index.json')

    # dump_to_file(idf_list,'idf_list.json')

    