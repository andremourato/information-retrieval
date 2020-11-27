###############################
#   Authors             
###############################
#   André Mourato nmec 84745
#   Gonçalo Marques nmec 80327
###############################
# Benchmarking
import time
import tracemalloc
# Necessary imports
import operator
import sys
import math
# File imports
from utils import *

def bm25_scoring(term_document_weights, document_terms, queries):
    '''Calculates bm25 scores for each query and returns a dict with each queries highest ranking documents, in descending order
    ----------
    term_document_weights : dict
        Dictionary that contains the token as the key and the number of occurences as the value.

    document_terms : dict
        Dictionary that contains the docID as the key and the list of terms contained in the document as the value.

    queries : list
        List of queries, in which each element is a list of the tokens of each query
        Example: [
            ['coronavirus', 'origin'],
            ['coronavirus', 'immunity']
        ]
        
    Returns
    -------
    scores : dict
        Dictionary of dictionaries that contains the query as the key 
        and a dictionary with the docIDs in which the query terms exist and corresponding bmc score, as the value.
        Example :{
            "1": {
                "ne5r4d4b": 3.418630282579524,
                "mv3crcsh": 3.382283292874245,
                "es7q6c90": 3.366348323271933,
                "zy8qjaai": 3.31746302919451,
            }
        }

    latencies : dict
        Dictionary that contains the query as the key and the latency in seconds as the value.
        Example :{
            "1": 0.1641572300000007,
            "2": 0.2601559629999999,
            "3": 0.1559371460000003,
            "4": 0.17825443500000038
        }
    '''
    scores = {}
    latencies = {}
    idx = 1
    for query in queries:
        query_latency_start = time.process_time()
        scores[idx] = {}
        # 1 - Calculates the score of each document for each query
        for docID in document_terms:
            scores[idx][docID] = 0
            for token in query:
                if token in document_terms[docID]:
                    scores[idx][docID] += term_document_weights[token][docID]
        scores[idx] = dict(sorted(scores[idx].items(), key=operator.itemgetter(1), reverse=True))
        latencies[idx] = time.process_time() - query_latency_start
        idx += 1
    return scores, latencies

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'bmc_weights.csv'
    else:
        filename = sys.argv[1]
    print('Loading weights from',filename)
    print('------------------------------------------------------------')
    print('STARTING BM25 RANKING...')
    print('------------------------------------------------------------')

    #########################################################
    # LOADING INFORMATION FROM FILES
    #########################################################
    # 1 - Loading the stopwords
    stopwords = load_stop_words('resources/stopwords.txt')
    # 2 - Loading the queries
    queries = load_queries('resources/queries.txt',stopwords)
    # 3 - Loads the weights that were previously calculated
    term_document_weights, document_terms, idf_list = load_weights(filename)

    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
    tracemalloc.start()
    time_start = time.process_time()
    #########################################################
    # RANKING
    #########################################################
    scores, latencies = bm25_scoring(term_document_weights, document_terms, queries)

    #########################################################
    # BENCHMARKING INFORMATION
    #########################################################
    time_elapsed = time.process_time() - time_start
    print('Total ranking time:',time_elapsed,'s')
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Memory usage for ranking was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    print('------------------------------------------------------------')

    #########################################################
    # CALCULATING METRICS (precision, recall, f_measure, average_precision, ndcg, latency)
    #########################################################
    results, query_throughput, median_latency, means  = calculate_metrics(scores,latencies,time_elapsed)

    #########################################################
    # DUMPING DATA STRUCTURES TO A FILE
    #########################################################
    dump_results('bmc_results.csv', results, query_throughput, median_latency, means, latencies)

    # dump_to_file(latencies, 'latencies.json')