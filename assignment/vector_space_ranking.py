# Benchmarking
import time
import tracemalloc
# Necessary imports
import operator
import sys
# File imports
from utils import *

def ltc_calculation(term_document_weights,document_terms,idf_list,queries):
    # 1 - COUNTS THE FREQUENCY OF TERMS IN EACH QUERY
    query_term_frequency = {}
    i = 1
    for query in queries:
        query_term_frequency[i] = {}
        for token in query:
            if token not in query_term_frequency[i]:
                query_term_frequency[i][token] = 1
            else:
                query_term_frequency[i][token] += 1
        i += 1
            
    # 2 - CALCULATES LTC AND SCORES
    query_term_weights = {}
    scores = {}
    latencies = {}
    for idx,query in enumerate(queries):
        query_latency_start = time.process_time()
        query_term_weights[idx+1] = {}
        # a - NON-NORMALIZED WEIGHT CALCULATION
        for token in query:
            # Calculates l * t = (1 + log10(tf)) * idf
            query_term_weights[idx+1][token] = ( 1 + math.log10(query_term_frequency[idx+1][token]) ) * idf_list[token]

        # b - CALCULATION OF THE NORM FACTOR
        norm_factor = 1/math.sqrt(sum([w**2 for w in query_term_weights[idx+1].values()]))

        # c - NORMALIZED WEIGHT CALCULATION
        for token in query:
            query_term_weights[idx+1][token] *= norm_factor
        
        # 3 - Score calculation ltc*lnc
        scores[idx+1] = {}
        for docID in document_terms:
            scores[idx+1][docID] = 0
            for token in query:
                if token not in document_terms[docID]:
                    continue
                scores[idx+1][docID] += query_term_weights[idx+1][token] * term_document_weights[token][docID]

        scores[idx+1] = dict(sorted(scores[idx+1].items(), key=operator.itemgetter(1), reverse=True))
        latencies[idx+1] = time.process_time() - query_latency_start

    return query_term_weights, scores, latencies

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

    query_term_weights, scores, latencies = weighting_tf_idf(term_document_weights,document_terms,idf_list,queries)

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

    dump_to_file(query_term_weights,'query_term_weights.json')

    dump_to_file(scores,'ranked_scores.json')
    
    dump_to_file(results,'results.json')

    dump_to_file(latencies, 'latencies.json')
