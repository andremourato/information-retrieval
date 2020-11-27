###############################
#   Authors             
###############################
#   André Mourato nmec 84745
#   Gonçalo Marques nmec 80327
###############################
import Stemmer
import json
import os
import math
import operator
import statistics 

QUERIES_RELEVANCE_FILTERED_FILE = 'resources/queries.relevance.filtered.txt'
OUTPUT_DIR = 'outputs/'
DEBUG_DIR = 'debug/'

#########################################################
# AUXILIAR METHODS
#########################################################
def remove_non_alpha(string):
    '''Removes all non-alpha characters from a string and returns a list of the individual extracted tokens
    ----------
    string : string
        The input string to parse
        Example: 'covid-19abc'
            
        
    Returns
    -------
    tokens : list
        The list of extracted tokens after removing all non-alpha characters
        Example: remove_non_alpha('covid-19abc') would return ['covid', 'abc']
    '''
    return ''.join([ c if c.isalpha() else ' ' for c in string.lower()]).split()

def calculate_status(engine_relevance,file_relevance):
    '''Checks if a document is a true positive, false positive, true negative or a false negative
    ----------
    engine_relevance : boolean
        True if the search engine considers the document relevant and False otherwise

    file_relevance : boolean
        True if the document is considered relevant by the gold standard and False otherwise
        
    Returns
    -------
    status : string
        The status of the document. Can be one of the following values: 'tp', fp', 'fn', 'tn'
    '''
    # If the engine considers the document RELEVANT
    if engine_relevance: 
        # If the input file considers the document RELEVANT
        if file_relevance > 0:# FILE RELEVANT
            return 'tp'
        else: # FILE NOT RELEVANT
            return 'fp'
    else: # ENGINE NOT RELEVANT
        # If the input file considers the document RELEVANT
        if file_relevance > 0:# FILE RELEVANT
            return 'fn'
        else: # FILE NOT RELEVANT
            return 'tn'

#########################################################
# METRIC CALCULATION
#########################################################
def calculate_metrics(scores, latencies, time_elapsed):
    '''Receives the document rankings for each query in the score dictionary, the latency of each query and total time elapsed for the ranking process.
       Compares the highest scoring documents with the list of relevant documents for each query to calculate evaluation metrics such as precision,
       f measure and normalized discounted cumulative gain.
       Finally calculates de mean values of each parameter for each query.
    ----------
    scores : dict
        Dictionary of dictionaries that contains the query as the key 
        and a dictionary with the docIDs in which the query terms exist and corresponding score, as the value.

    latencies : dict
        Dictionary that contains the query as the key and the latency in seconds as the value.

    time_elapsed : float
        Total time count used by the scoring process.
        
    Returns
    -------
    results : dict
        Dictionary of dictionaries containing the calculated metrics for each query
        for diferent result retrieval windows
        Example: {
            "1": {
                "10": {
                    "tp": 3,
                    "fp": 5,
                    "fn": 207,
                    "tn": 245,
                    "avg_precision": 0.0007384585289514867,
                    "precision": 0.375,
                    "recall": 0.014285714285714285,
                    "fmeasure": 0.027522935779816515
                }
            }    
        }

    query_throughput : float
        Number of queries processed per second

    median_latency : float
        Median of the amount of time one must after issuing a query 
        before receiving a response

    means : dict
        Dictionary containing the mean value for all queries of each metric
        Example: {
            "precision10": 0.384,
            "precision20": 0.321,
            "precision50": 0.22240000000000001,
            "recall10": 0.07991754131744881,
            "recall20": 0.13456002463048145,
            "recall50": 0.20650537292465992
        }

    '''
    # Results contains tp, fp, fn, tn of each query and other metrics
    results = {}
    means = {}
    relevance = load_query_relevance()        
    for query in scores:
        # Used to count the number of tp, fp, fn and tn
        #  in the top10, top20 and top50 of each query
        results[query] = {
            10:{
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'tn': 0,
            },
            20:{
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'tn': 0,
            },
            50:{
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'tn': 0,
            }
        }
        # Number of relevant documents at every step
        top10_num_relevant = 0
        top20_num_relevant = 0
        top50_num_relevant = 0

        # AVERAGES
        results[query][10]['avg_precision'] = 0
        results[query][20]['avg_precision'] = 0
        results[query][50]['avg_precision'] = 0

        # DCG - Discounted Cumulative Gain
        ideal_dcg = {
            10: 0,
            20: 0,
            50: 0
        }
        idx = 1
        # Calculates the ideal DCG
        for doc_id in relevance[query]:
            if idx == 1:
                ideal_dcg[10] += relevance[query][doc_id]
                ideal_dcg[20] += relevance[query][doc_id]
                ideal_dcg[50] += relevance[query][doc_id]
            else:
                if idx <= 10:
                    ideal_dcg[10] += relevance[query][doc_id] / math.log2(idx)
                if idx <= 20:
                    ideal_dcg[20] += relevance[query][doc_id] / math.log2(idx)
                if idx <= 50:
                    ideal_dcg[50] += relevance[query][doc_id] / math.log2(idx)
            idx += 1
        # Used to store the real DCG
        dcg10 = 0
        dcg20 = 0
        dcg50 = 0

        docs = list(scores[query].keys())
        for i,doc_id in enumerate(docs):
            # Documents that don't appear in this query in the file are NOT RELEVANT
            # file_relevant can be 0, 1 or 2 according to the relevance of a given document and for a given query
            if doc_id not in relevance[query]:
                file_relevant = 0
            else:
                file_relevant = relevance[query][doc_id]

            is_top10 = False
            is_top20 = False
            is_top50 = False
            
            # If the current document is in the Top 10
            if i < 10:
                is_top10 = True
                dcg10 += file_relevant if i == 0 else file_relevant/math.log2(i+1)
                # Calculates average precision
                if file_relevant > 0:
                    top10_num_relevant += 1
                    results[query][10]['avg_precision'] += top10_num_relevant / (i+1)
            # If the current document is in the Top 20
            if i < 20:
                is_top20 = True
                dcg20 += file_relevant if i == 0 else file_relevant/math.log2(i+1)
                # Calculates average precision
                if file_relevant > 0:
                    top20_num_relevant += 1
                    results[query][20]['avg_precision'] += top20_num_relevant / (i+1)
            # If the current document is in the Top 50
            if i < 50:
                is_top50 = True
                dcg50 += file_relevant if i == 0 else file_relevant/math.log2(i+1)
                # Calculates average precision
                if file_relevant > 0:
                    top50_num_relevant += 1
                    results[query][50]['avg_precision'] += top50_num_relevant / (i+1)

            # Decides if it is a tp, tn, fp, fn
            results[query][10][calculate_status(is_top10,file_relevant)] += 1
            results[query][20][calculate_status(is_top20,file_relevant)] += 1
            results[query][50][calculate_status(is_top50,file_relevant)] += 1
        
        ##################################
        # 1 - PRECISION | P = tp/(tp + fp)
        ##################################
        top10_den = results[query][10]['tp'] + results[query][10]['fp']
        top20_den = results[query][20]['tp'] + results[query][20]['fp']
        top50_den = results[query][50]['tp'] + results[query][50]['fp']
        # Top 10
        if top10_den != 0:
            results[query][10]['precision'] = results[query][10]['tp'] / top10_den
        else:
            results[query][10]['precision'] = 0
        # Top 20
        if top20_den != 0:
            results[query][20]['precision'] = results[query][20]['tp'] / top20_den
        else:
            results[query][20]['precision'] = 0
        # Top 50
        if top50_den != 0:
            results[query][50]['precision'] = results[query][50]['tp'] / top50_den
        else:
            results[query][50]['precision'] = 0
        
        ##################################
        # 2 - RECALL | R = tp/(tp + fn)
        ##################################
        top10_den = results[query][10]['tp'] + results[query][10]['fn']
        top20_den = results[query][20]['tp'] + results[query][20]['fn']
        top50_den = results[query][50]['tp'] + results[query][50]['fn']
        # Top 10
        if top10_den != 0:
            results[query][10]['recall'] = results[query][10]['tp'] / top10_den
        else:
            results[query][10]['recall'] = 0
        # Top 20
        if top20_den != 0:
            results[query][20]['recall'] = results[query][20]['tp'] / top20_den
        else:
            results[query][20]['recall'] = 0
        # Top 50
        if top50_den != 0:
            results[query][50]['recall'] = results[query][50]['tp'] / top50_den
        else:
            results[query][50]['recall'] = 0

        ##################################
        # 3 - F MEASURE | F = 2RP/(R+P)
        ##################################
        top10_den = results[query][10]['recall'] + results[query][10]['precision']
        top20_den = results[query][20]['recall'] + results[query][20]['precision']
        top50_den = results[query][50]['recall'] + results[query][50]['precision']
        # Top 10
        if top10_den != 0:
            results[query][10]['fmeasure'] = 2 * results[query][10]['recall'] * results[query][10]['precision'] / top10_den
        else:
            results[query][10]['fmeasure'] = 0
        # Top 20
        if top20_den != 0:
            results[query][20]['fmeasure'] = 2 * results[query][20]['recall'] * results[query][20]['precision'] / top20_den
        else:
            results[query][20]['fmeasure'] = 0
        # Top 50
        if top50_den != 0:
            results[query][50]['fmeasure'] = 2 * results[query][50]['recall'] * results[query][50]['precision'] / top50_den
        else:
            results[query][50]['fmeasure'] = 0
        
        ##################################
        # 4 - Average Precision
        ##################################
        # Top 10
        if top10_num_relevant != 0:
            results[query][10]['avg_precision'] = results[query][10]['avg_precision'] / top10_num_relevant
        else:
            results[query][10]['avg_precision'] = 0
        # Top 20
        if top20_num_relevant != 0:
            results[query][20]['avg_precision'] = results[query][20]['avg_precision'] / top20_num_relevant
        else:
            results[query][20]['avg_precision'] = 0
        # Top 50
        if top50_num_relevant != 0:
            results[query][50]['avg_precision'] = results[query][50]['avg_precision'] / top50_num_relevant
        else:
            results[query][50]['avg_precision'] = 0

        ##################################
        # 5 - NDCG
        ##################################
        # Top 10
        if ideal_dcg[10] != 0:
            results[query][10]['ndcg'] = dcg10 / ideal_dcg[10]
        else:
            results[query][10]['ndcg'] = 0
        # Top 20
        if ideal_dcg[20] != 0:
            results[query][20]['ndcg'] = dcg20 / ideal_dcg[20]
        else:
            results[query][20]['ndcg'] = 0
        # Top 50
        if ideal_dcg[50] != 0:
            results[query][50]['ndcg'] = dcg50 / ideal_dcg[50]
        else:
            results[query][50]['ndcg'] = 0

    ##################################
    # 7 - Query Throughput
    ##################################
    query_throughput = len(scores) / time_elapsed

    ##################################
    # 8 - Median query latency
    ##################################
    median_latency = statistics.median(latencies.values())
    
    ##################################
    # 9 Mean Values
    ##################################
    means['precision10'] = statistics.mean([ query[10]['precision'] for query in results.values()])
    means['precision20'] = statistics.mean([ query[20]['precision'] for query in results.values()])
    means['precision50'] = statistics.mean([ query[50]['precision'] for query in results.values()])
    means['recall10'] = statistics.mean([ query[10]['recall'] for query in results.values()])
    means['recall20'] = statistics.mean([ query[20]['recall'] for query in results.values()])
    means['recall50'] = statistics.mean([ query[50]['recall'] for query in results.values()])
    means['fmeasure10'] = statistics.mean([ query[10]['fmeasure'] for query in results.values()])
    means['fmeasure20'] = statistics.mean([ query[20]['fmeasure'] for query in results.values()])
    means['fmeasure50'] = statistics.mean([ query[50]['fmeasure'] for query in results.values()])
    means['map10'] = statistics.mean([ query[10]['avg_precision'] for query in results.values()])
    means['map20'] = statistics.mean([ query[20]['avg_precision'] for query in results.values()])
    means['map50'] = statistics.mean([ query[50]['avg_precision'] for query in results.values()])
    means['ndcg10'] = statistics.mean([ query[10]['ndcg'] for query in results.values()])
    means['ndcg20'] = statistics.mean([ query[20]['ndcg'] for query in results.values()])
    means['ndcg50'] = statistics.mean([ query[50]['ndcg'] for query in results.values()])

    return results, query_throughput, median_latency, means

#########################################################
# FILE METHODS
#########################################################
def load_stop_words(file):
    '''Loads the list of stop words from a file
    ----------
    file : string
        The file that contains the stop words, separated by the newline character            
        
    Returns
    -------
    stopwords : list
        The list of stopwords
    '''
    with open(file)  as f_in:
        return [ _.split()[0] for _ in f_in ]

def load_queries(file,stopwords):
    '''Loads the list of queries from a file and tokenizes each term
    ----------
    file : string
        The file that contains the stop words, separated by the newline character   
        
    stopwords : list
        The list of stopwords          
        
    Returns
    -------
    queries : list
        List of queries, in which each element is a list of the tokens of each query
        Example: [
            ['coronavirus', 'origin'],
            ['coronavirus', 'immunity']
        ]
    '''
    with open(file)  as f_in:
        return [
            Stemmer.Stemmer('porter').stemWords(\
                remove_non_alpha(' '.join(\
                    [word for word in q.split() if len(word) >= 3 and word not in stopwords]))) for q in f_in
        ]

def load_query_relevance():
    '''Loads the list of queries from a file and tokenizes each term
    ----------
        
    Returns
    -------
    relevances : dict
        Dictionary that associates the relevance of each document for a given query, 
        according to the gold standard
        Example: 
        {
            "1": {
                "010vptx3": 2,
                "0be4wta5": 2,
                "1abp6oom": 1,
                "1bvsn9e8": 0
            }
        }
    '''
    relevance = {}
    with open(QUERIES_RELEVANCE_FILTERED_FILE)  as f_in:
        for line in f_in.readlines():
            query,doc_id,rel = line.split()
            query = int(query)
            rel = int(rel)
            if query not in relevance:
                relevance[query] = {}
            relevance[query][doc_id] = rel
        for query in relevance:
            relevance[query] = dict(sorted(relevance[query].items(), key=operator.itemgetter(1), reverse=True))
    return relevance

def load_weights(filename):
    '''Loads the list of queries from a file and tokenizes each term
    ----------
    filename : string
        The file that contains the weights to be read

    Returns
    -------
    term_document_weights : dict
        Dictionary that contains the token as the key and the number of occurences as the value.
        
    document_terms : dict
        Dictionary that contains the docID as the key and the list of terms contained in the document as the value.

    idf_list : dict
        Dictionary that contains the token as the key and the idf as the value.
    '''
    term_document_weights = {}
    document_terms = {}
    idf_list = {}
    with open("%s%s" % (OUTPUT_DIR,filename)) as f_in:
        for line in f_in.readlines():
            # Processes each line
            tmp = line.strip().split(';')
            # Processes the term and associates its idf
            term,idf = tmp[0].split(':')
            idf_list[term] = float(idf)
            # Processes the weight of the term in each of the documents
            for doc in tmp[1:]:
                doc_id, doc_weight = doc.split(':')
                if doc_id not in document_terms:
                    document_terms[doc_id] = []
                document_terms[doc_id].append(term)
                if term not in term_document_weights:
                    term_document_weights[term] = {}
                term_document_weights[term][doc_id] = float(doc_weight)
    return term_document_weights, document_terms, idf_list

def dump_to_file(dic,filename):
    '''Writes a dictionary to a file in the JSON format
    ----------
    dic : dict
        Any dictionary

    filename : string
        The file to where the dict should be written
    '''
    if not os.path.exists(DEBUG_DIR):
        os.mkdir(DEBUG_DIR, 0o775)
    print('DUMPING TO FILE %s' % filename)
    with open("%s%s" % (DEBUG_DIR,filename), "w") as write_file:
        json.dump(dic, write_file, indent=4)
    
def dump_weights(term_document, idf_list, filename):
    '''Writes the term idfs and weights to a file
    ----------
    term_document : dict
        Dictionary that contains the token as the key and the number of occurences as the value.

    idf_list : dict
        Dictionary that contains the token as the key and the idf as the value.

    filename : string
        The file to where the dict should be written
    '''
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR, 0o775)
    with open("%s%s" % (OUTPUT_DIR,filename), "w") as write_file:
        for (token,idf) in idf_list.items():
            s = '%s:%.15f' % (token,idf)
            for docID in term_document[token]:
                s += ';%s:%.15f' % (docID,term_document[token][docID])
            write_file.write("%s\n" % s)

def dump_results(file_out, results, query_throughput, median_latency, means, latencies):
    '''Writes the results to a file
    ----------
    file_out : string
        The file to where the results should be written
        
    results : dict
        Dictionary of dictionaries containing the calculated metrics for each query
        for diferent result retrieval windows
    
    query_throughput : float
        Number of queries processed per second

    median_latency : float
        Median of the amount of time one must after issuing a query 
        before receiving a response

    means : dict
        Dictionary containing the mean value for all queries of each metric

    latencies : dict
        Dictionary that contains the query as the key and the latency in seconds as the value.

    '''
    with open("%s%s" % (OUTPUT_DIR,file_out), "w") as write_file:
        write_file.write('query;precision10;precision20;precision50;recall10;recall20;recall50;fmeasure10;fmeasure20;fmeasure50;avgprecision10;avgprecision20;avgprecision50;ndcg10;ndcg20;ndcg50;latency\n')
        idx = 1
        for query in results:
            s = str(idx) + ';'
            # Precision
            s += '%f;%f;%f;' % (results[query][10]['precision'],
                                results[query][20]['precision'],
                                results[query][50]['precision'])

            # Recall  
            s += '%f;%f;%f;' % (results[query][10]['recall'],
                                results[query][20]['recall'],
                                results[query][50]['recall'])

            # F Measure
            s += '%f;%f;%f;' % (results[query][10]['fmeasure'],
                                results[query][20]['fmeasure'],
                                results[query][50]['fmeasure'])

            # Avg Precision
            s += '%f;%f;%f;' % (results[query][10]['avg_precision'],
                                results[query][20]['avg_precision'],
                                results[query][50]['avg_precision'])

            # NDCG
            s += '%f;%f;%f;' % (results[query][10]['ndcg'],
                                results[query][20]['ndcg'],
                                results[query][50]['ndcg'])
            # latency
            s += str(latencies[idx]) + '\n'
            idx += 1
            write_file.write(s)
        # Writes the last line with the mean values of each metric (and writes the median of the latencies)
        write_file.write(
            'mean;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f' % 
            (means['precision10'],
            means['precision20'],
            means['precision50'],
            means['recall10'],
            means['recall20'],
            means['recall50'],
            means['fmeasure10'],
            means['fmeasure20'],
            means['fmeasure50'],
            means['map10'],
            means['map20'],
            means['map50'],
            means['ndcg10'],
            means['ndcg20'],
            means['ndcg50'],
            median_latency,
            query_throughput))