import Stemmer
import json
import os
import math
import operator
import statistics 

QUERIES_RELEVANCE_FILTERED_FILE = 'resources/queries.relevance.filtered.txt'
INDEXER_OUTPUT_DIR = 'outputs/'
DEBUG_DIR = 'debug/'
VECTOR_SPACE_RANKING_RESULT_FILE = 'outputs/'

#########################################################
# AUXILIAR METHODS
#########################################################
def remove_non_alpha(string):
    return ''.join([ c if c.isalpha() else ' ' for c in string.lower()]).split()

def calculate_status(engine_relevance,file_relevance):
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
    # Results contains tp, fp, fn, tn of each query and other metrics
    results = {}
    means = {}
    relevance = load_query_relevance()        
    for query in scores:
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
        #number of relevant documents at every step
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
        
        dcg10 = 0
        dcg20 = 0
        dcg50 = 0

        docs = list(scores[query].keys())
        for i,doc_id in enumerate(docs):
            # Documents that don't appear in this query in the file are NOT RELEVANT
            # Relevance can be 0, 1 or 2
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
    with open(file)  as f_in:
        return [ _.split()[0] for _ in f_in ]

def load_queries(file,stopwords):
    with open(file)  as f_in:
        return [
            Stemmer.Stemmer('porter').stemWords(\
                remove_non_alpha(' '.join(\
                    [word for word in q.split() if len(word) >= 3 and word not in stopwords]))) for q in f_in
        ]

def load_query_relevance():
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
        dump_to_file(relevance,'relevance_sorted.json')
    return relevance

def load_weights(filename):
    term_document_weights = {}
    document_terms = {}
    idf_list = {}
    with open("%s%s" % (INDEXER_OUTPUT_DIR,filename)) as f_in:
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
    if not os.path.exists(DEBUG_DIR):
        os.mkdir(DEBUG_DIR, 0o775)
    print('DUMPING TO FILE %s' % filename)
    with open("%s%s" % (DEBUG_DIR,filename), "w") as write_file:
        json.dump(dic, write_file, indent=4)
    
def dump_weights(term_document, idf_list, filename):
    with open("%s%s" % (INDEXER_OUTPUT_DIR,filename), "w") as write_file:
        for (token,idf) in idf_list.items():
            s = '%s:%.15f' % (token,idf)
            for docID in term_document[token]:
                s += ';%s:%.15f' % (docID,term_document[token][docID])
            write_file.write("%s\n" % s)

def dump_results(file_out, results, query_throughput, median_latency, means, latencies):
    with open("%s%s" % (VECTOR_SPACE_RANKING_RESULT_FILE,file_out), "w") as write_file:
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
            'mean;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f' % 
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
            median_latency))