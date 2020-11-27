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
# File imports
from utils import *

def scoring_tf_idf(term_document_weights,document_terms,idf_list,queries):
    '''Counts term frequency and calculates ltc normalized weight for each query.
       Afterwards calculates the lnc.ltc score of each document for each query.
    ----------
    term_document_weights : dict
        Dictionary that contains the token as the key and the number of occurences as the value.

    document_terms : dict
        Dictionary that contains the docID as the key and the list of terms contained in the document as the value.

    idf_list : dict
        Dictionary that contains the token as the key and the idf as the value.

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
        and a dictionary with the docIDs in which the query terms exist and corresponding tf idf score, as the value.
        Example :{
            "1": {
                "9dj07sac": 0.4220587521433612,
                "ezi2mret": 0.40037289998178693,
                "e3wjo0yk": 0.396389326695161,
                "t7gpi2vo": 0.3961214403104035
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
    scores = {}
    latencies = {}
    for idx,query in enumerate(queries):
        query_latency_start = time.process_time()
        query_term_weights = {}
        # a - NON-NORMALIZED WEIGHT CALCULATION
        for token in query:
            # Calculates l * t = (1 + log10(tf)) * idf
            query_term_weights[token] = ( 1 + math.log10(query_term_frequency[idx+1][token]) ) * idf_list[token]

        # b - CALCULATION OF THE NORM FACTOR
        norm_factor = 1/math.sqrt(sum([w**2 for w in query_term_weights.values()]))

        # c - NORMALIZED WEIGHT CALCULATION
        for token in query:
            query_term_weights[token] *= norm_factor
        
        # 3 - Score calculation ltc*lnc
        scores[idx+1] = {}
        for docID in document_terms:
            scores[idx+1][docID] = 0
            for token in query:
                if token in document_terms[docID]:
                    scores[idx+1][docID] += query_term_weights[token] * term_document_weights[token][docID]

        scores[idx+1] = dict(sorted(scores[idx+1].items(), key=operator.itemgetter(1), reverse=True))
        latencies[idx+1] = time.process_time() - query_latency_start

    return scores, latencies

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'tf_idf_weights.csv'
    else:
        filename = sys.argv[1]
    print('Loading weights from',filename)
    print('------------------------------------------------------------')
    print('STARTING VECTOR SPACE RANKING...')
    print('------------------------------------------------------------')

    #########################################################
    # LOADING INFORMATION FROM FILES
    #########################################################
    # 1 - Loading the stopwords 
    stopwords = load_stop_words('resources/stopwords.txt')
    # 2 - Loading the queries
    queries = load_queries('resources/queries.txt',stopwords)
    # 3 - Loading the term weights and idfs
    term_document_weights, document_terms, idf_list = load_weights(filename)

    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
    tracemalloc.start()
    time_start = time.process_time()
    #########################################################
    # RANKING
    #########################################################
    scores, latencies = scoring_tf_idf(term_document_weights,document_terms,idf_list,queries)

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
    dump_results('vector_space_results.csv', results, query_throughput, median_latency, means, latencies)
    
    # dump_to_file(document_terms,'document_terms.json')
    
    # dump_to_file(means,'means.json')

    # dump_to_file(term_document_weights,'ranked_term_document_weights.json')

    # dump_to_file(idf_list,'ranked_idf_list.json')

    # dump_to_file(scores,'ranked_scores.json')
    
    # dump_to_file(results,'results.json')

    # dump_to_file(latencies, 'latencies.json')
