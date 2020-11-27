# Benchmarking
import time
import tracemalloc
# Necessary imports
import operator
import sys
# File imports
from utils import *

def ltc_calculation(term_document_weights,document_terms,idf_list,queries):
    query_document_weights = {}
    document_query_weights = {}
    scores = {}
    latencies = {}
    for idx,query in enumerate(queries):
        query_latency_start = time.process_time()
        for token in query:
            # If the token exists in at least one document
            for docID in document_terms:
                # Query document
                if token not in query_document_weights:
                    query_document_weights[token] = {}
                # Document Query
                if docID not in document_query_weights:
                    document_query_weights[docID] = {}

                if token in document_terms[docID]:
                    query_document_weights[token][docID] = idf_list[token] * term_document_weights[token][docID]
                    document_query_weights[docID][token] = idf_list[token] * term_document_weights[token][docID]
                else:
                    query_document_weights[token][docID] = 0
                    document_query_weights[docID][token] = 0
        # 3 - Score calculation
        scores[idx+1] = { docID: sum(v.values()) for docID,v in document_query_weights.items() }
        scores[idx+1] = dict(sorted(scores[idx+1].items(), key=operator.itemgetter(1), reverse=True))
        latencies[idx+1] = time.process_time() - query_latency_start

    return query_document_weights, document_query_weights, scores, latencies  

def weighting_tf_idf(term_document_weights,document_terms,idf_list,queries):
    # 1 - LTC calculation
    return ltc_calculation(term_document_weights,document_terms,idf_list,queries)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'outputs/indexer_output.txt'
    else:
        filename = sys.argv[1]
    print('Loading weights from',filename)
    print('------------------------------------------------------------')
    print('STARTING RANKING...')
    print('------------------------------------------------------------')

    #########################################################
    # LOADING INFORMATION FROM FILES
    #########################################################
    # 1 - Loading the term weights and idfs
    term_document_weights, document_terms, idf_list = load_term_idf_weights()
    # 2 - Loading the stopwords 
    stopwords = load_stop_words('resources/stopwords.txt')
    # 3 - Loading the queries
    queries = load_queries('resources/queries.txt',stopwords)

    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
    tracemalloc.start()
    time_start = time.process_time()
    #########################################################
    # RANKING
    #########################################################

    query_document_weights, document_query_weights, scores, latencies = weighting_tf_idf(term_document_weights,document_terms,idf_list,queries)

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
    dump_to_file(means,'means.json')

    dump_results('vector_space_results.csv', results, query_throughput, median_latency, means, latencies)

    dump_to_file(term_document_weights,'ranked_term_document_weights.json')

    dump_to_file(idf_list,'ranked_idf_list.json')

    dump_to_file(scores,'ranked_scores.json')
    
    dump_to_file(results,'results.json')

    dump_to_file(document_query_weights, 'document_query_weights.json')

    dump_to_file(latencies, 'latencies.json')
