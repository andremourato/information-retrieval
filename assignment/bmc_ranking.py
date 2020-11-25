# Benchmarking
import time
import tracemalloc
# Necessary imports
import operator
import sys
import math
# File imports
from utils import *

def bm25_avdl(document_length_index):

    avdl = sum(v for v in document_length_index.values())

    return avdl / len(document_length_index)

def bm25_weighting(N, k, b, avdl, bmc_data, document_length_index, idf_list, queries):
    weights = {}
    scores = {}
    for idx,query in enumerate(queries):
        if idx+1 not in scores:
            scores[idx+1] = {} 

        for token in query:
            if token not in weights:
                weights[token] = {} 

                for docID in bmc_data[token]:
                    first = math.log10(N/idf_list[token])
                    second = (k+1) * bmc_data[token][docID]
                    third = 1 / (( k*((1-b) + (b*document_length_index[docID] / avdl)) + bmc_data[token][docID]))
                    weights[token][docID] = first*second*third 
                    if docID not in scores[idx+1]:
                        scores[idx+1][docID] = 0
                    scores[idx+1][docID] += first*second*third 

        scores[idx+1] = dict(sorted(scores[idx+1].items(), key=operator.itemgetter(1), reverse=True))

    return weights, scores

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

    # 1 - Loading document length
    with open('debug/document_length_index.json') as json_file:
        document_length_index = json.load(json_file)
    # 2 - Loading the stopwords
    stopwords = load_stop_words('resources/stopwords.txt')
    # 3 - Loading the queries
    queries = load_queries('resources/queries.txt',stopwords)
    #########################################################
    # RANKING
    #########################################################

    bmc_data, document_terms, idf_list = load_bmc()
    avdl = bm25_avdl(document_length_index)
    N = len(document_length_index)
    k = 1.2
    b = 0.75
    
    bmc_weights = {}
    bmc_weights, scores = bm25_weighting(N, k, b, avdl, bmc_data, document_length_index, idf_list, queries)


    #########################################################
    # BENCHMARKING INFORMATION
    #########################################################
    print('Total ranking time:',time.process_time() - time_start,'s')
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Memory usage for ranking was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    print('------------------------------------------------------------')

    #########################################################
    # CALCULATING METRICS (precision, recall, f_measure, average_precision, ndcg, latency)
    #########################################################
    results = calculate_metrics(scores)

    #########################################################
    # DUMPING DATA STRUCTURES TO A FILE
    #########################################################

    dump_to_file(bmc_weights, 'bmc_weights.json')
    dump_to_file(scores, 'bmc_scores.json')

    dump_to_file(results, 'bmc_results.json')