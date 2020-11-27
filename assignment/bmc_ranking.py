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
        filename = 'outputs/indexer_output.txt'
    else:
        filename = sys.argv[1]
    print('Loading weights from',filename)
    print('------------------------------------------------------------')
    print('STARTING RANKING...')
    print('------------------------------------------------------------')
    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
    tracemalloc.start()
    time_start = time.process_time()

    # 1 - Loading the stopwords
    stopwords = load_stop_words('resources/stopwords.txt')
    # 2 - Loading the queries
    queries = load_queries('resources/queries.txt',stopwords)
    # 3 - Loads the weights that were previously calculated
    term_document_weights, document_terms, idf_list = load_weights('bmc_weights.csv')

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
    dump_to_file(scores, 'bmc_scores.json')

    dump_results('bmc_results.csv', results, query_throughput, median_latency, means, latencies)