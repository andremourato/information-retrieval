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
            9dj07sac: {
                "incub": 4,
                "period": 4,
                "epidemiolog": 2,
                "characterist": 1,
                "novel": 2,
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

def ltc_calculation(term_document_weights,document_term_weights,idf_list,queries):
    query_document_weights = {}
    document_query_weights = {}
    scores = {}
    for idx,query in enumerate(queries):
        for token in query:
            # If the token exists in at least one document
            for docID in document_term_weights:
                # Query document
                if token not in query_document_weights:
                    query_document_weights[token] = {}
                # Document Query
                if docID not in document_query_weights:
                    document_query_weights[docID] = {}

                if token in document_term_weights[docID]:
                    query_document_weights[token][docID] = idf_list[token] * term_document_weights[token][docID]
                    document_query_weights[docID][token] = idf_list[token] * term_document_weights[token][docID]
                else:
                    query_document_weights[token][docID] = 0
                    document_query_weights[docID][token] = 0
        # 3 - Score calculation
        scores[idx] = { docID: sum(v.values()) for docID,v in document_query_weights.items() }
        scores[idx] = dict(sorted(scores[idx].items(), key=operator.itemgetter(1), reverse=True))

    return query_document_weights, document_query_weights, scores

def dump_term_idf_weights_to_file(document_term_weights, term_document_weights, idf_list):
    with open("index_tf_idf.txt", "w") as write_file:
        #json.dump(term_document_weights, write_file, indent=4)
        for (token,idf) in idf_list.items():
            s = '%s:%f' % (token,idf)
            for docID in term_document_weights[token]:
                s += ';%s:%f' % (docID,term_document_weights[token][docID])
            write_file.write("%s\n" % s)

def weighting_tf_idf(term_index,document_length_index,queries):
    # 1 - LNC calculation
    document_term_weights, term_document_weights, idf_list = lnc_calculation(term_index,document_length_index)

    dump_term_idf_weights_to_file(document_term_weights, term_document_weights, idf_list)

    # 2 - LTC calculation
    query_weights, document_query_weights, scores = ltc_calculation(term_document_weights,document_term_weights,idf_list,queries)
    
    return document_term_weights, term_document_weights, query_weights, document_query_weights, scores, idf_list

# def weighting_bm25(term_index,document_length_index):
#     weights = {}

#     for docID in term_index:
#         for token in term_index[docID]:
#             if docID not in weights:
#                 weights[docID] = {}
#             weights[docID][token] = 1 + math.log10(term_index[docID][token])

#         norm_factor = 1/math.sqrt(sum([weights[docID][_]**2 for _ in weights[docID]]))

#         for token in weights[docID]:
#             weights[docID][token] *= norm_factor
#     return weights

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'datasets/metadata_2020-03-27.csv'
    else:
        filename = sys.argv[1]
    print('Reading dataset from file',filename)
    print('------------------------------')

    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
    tracemalloc.start()
    time_start = time.process_time()

    #########################################################
    # LOADING INFORMATION FROM FILE
    #########################################################
    stopwords = load_stop_words('stopwords.txt')
    queries = load_queries('queries.txt',stopwords)

    #########################################################
    # INDEXER
    #########################################################
    term_index, document_length_index = indexer(filename)
    
    # weights = weighting_bm25(term_index,document_length_index,queries)
    document_term_weights, term_document_weights,query_weights, document_query_weights, scores, idf_list = weighting_tf_idf(term_index,document_length_index,queries)
    
    #########################################################
    # BENCHMARKING INFORMATION
    #########################################################
    print('a) Total indexing time:',time.process_time() - time_start,'s')
    current, peak = tracemalloc.get_traced_memory()
    print(f"a) Memory usage for indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()

    print('\nb) Total vocabulary size is: ',len(term_index),'words')

    #########################################################
    # DUMPING DATA STRUCTURES TO A FILE
    #########################################################
    # dump_to_file(term_index,'term_index.json')

    # dump_to_file(document_term_weights,'document_term_weights.json')

    dump_to_file(term_document_weights,'term_document_weights.json')
    
    # dump_to_file(document_length_index,'document_length_index.json')

    dump_to_file(idf_list,'idf_list.json')
        
    dump_to_file(query_weights,'query_weights.json')

    # dump_to_file(document_query_weights,'document_query_weights.json')

    dump_to_file(scores,'scores.json')
    