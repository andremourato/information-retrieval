import Stemmer
import json
import os

INDEXER_OUTPUT_FILE = 'outputs/indexer_output.txt'
BMC_OUTPUT_FILE = 'outputs/bmc_data.txt'
DEBUG_DIR = 'debug/'
QUERIES_RELEVANCE_FILTERED_FILE = 'resources/queries.relevance.filtered.txt'

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
def calculate_metrics(scores):
    # results contains tp, fp, fn, tn of each query
    results = {}
    relevance = load_query_relevance()        
    mean_avg_precision10 = 0
    mean_avg_precision20 = 0
    mean_avg_precision50 = 0
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

        #temporary precision
        top10_temp_precision = 0
        top20_temp_precision = 0
        top50_temp_precision = 0

        docs = list(scores[query].keys())
        for i,doc_id in enumerate(docs):
            # Skips documents that don't appear in this query in the file
            if doc_id not in relevance[query]:
                continue

            # Can be 0, 1 or 2
            file_relevant = relevance[query][doc_id]
            top10_state = calculate_status(True if i < 10 else False,file_relevant)
            top20_state = calculate_status(True if i < 20 else False,file_relevant)
            top50_state = calculate_status(True if i < 50 else False,file_relevant)
            
            # Calculates average precision
            if i < 10:
                top10_num_relevant += 1
            if i < 20:
                top20_num_relevant += 1
            if i < 50:
                top50_num_relevant += 1

            top10_temp_precision = top10_num_relevant / (i+1)
            top20_temp_precision = top20_num_relevant / (i+1)
            top50_temp_precision = top50_num_relevant / (i+1)

            if i < 10:
                results[query][10]['avg_precision'] += top10_num_relevant / (i+1)
            if i < 20:
                results[query][20]['avg_precision'] += top20_num_relevant / (i+1)
            if i < 50:
                results[query][50]['avg_precision'] += top50_num_relevant / (i+1)

            results[query][10][top10_state] += 1
            results[query][20][top20_state] += 1
            results[query][50][top50_state] += 1
            
        # 1 - PRECISION | P = tp/(tp + fp)
        top10_den = results[query][10]['tp'] + results[query][10]['fp']
        top20_den = results[query][20]['tp'] + results[query][20]['fp']
        top50_den = results[query][50]['tp'] + results[query][50]['fp']
        if top10_den != 0:
            results[query][10]['precision'] = results[query][10]['tp'] / top10_den
        else:
            results[query][10]['precision'] = 0
        if top20_den != 0:
            results[query][20]['precision'] = results[query][20]['tp'] / top20_den
        else:
            results[query][20]['precision'] = 0
        if top50_den != 0:
            results[query][50]['precision'] = results[query][50]['tp'] / top50_den
        else:
            results[query][50]['precision'] = 0
        
        # 2 - RECALL | R = tp/(tp + fn)
        top10_den = results[query][10]['tp'] + results[query][10]['fn']
        top20_den = results[query][20]['tp'] + results[query][20]['fn']
        top50_den = results[query][50]['tp'] + results[query][50]['fn']
        if top10_den != 0:
            results[query][10]['recall'] = results[query][10]['tp'] / top10_den
        else:
            results[query][10]['recall'] = 0
        if top20_den != 0:
            results[query][20]['recall'] = results[query][20]['tp'] / top20_den
        else:
            results[query][20]['recall'] = 0
        if top50_den != 0:
            results[query][50]['recall'] = results[query][50]['tp'] / top50_den
        else:
            results[query][50]['recall'] = 0

        # 3 - F MEASURE | F = 2RP/(R+P)
        top10_den = results[query][10]['recall'] + results[query][10]['precision']
        top20_den = results[query][20]['recall'] + results[query][20]['precision']
        top50_den = results[query][50]['recall'] + results[query][50]['precision']
        if top10_den != 0:
            results[query][10]['fmeasure'] = 2 * results[query][10]['recall'] * results[query][10]['precision'] / top10_den
        else:
            results[query][10]['fmeasure'] = 0
        if top20_den != 0:
            results[query][20]['fmeasure'] = 2 * results[query][20]['recall'] * results[query][20]['precision'] / top20_den
        else:
            results[query][20]['fmeasure'] = 0
        if top50_den != 0:
            results[query][50]['fmeasure'] = 2 * results[query][50]['recall'] * results[query][50]['precision'] / top50_den
        else:
            results[query][50]['fmeasure'] = 0
        
        # 4 - Average Precision
        results[query][10]['avg_precision'] = results[query][10]['avg_precision'] / len(docs)
        results[query][20]['avg_precision'] = results[query][20]['avg_precision'] / len(docs)
        results[query][50]['avg_precision'] = results[query][50]['avg_precision'] / len(docs)

        mean_avg_precision10 += results[query][10]['avg_precision']
        mean_avg_precision20 += results[query][20]['avg_precision']
        mean_avg_precision50 += results[query][50]['avg_precision']
        print(query)
        print(results[query][10]['avg_precision'])
        print(results[query][20]['avg_precision'])
        print(results[query][50]['avg_precision'])
        print('\n')
    
    mean_avg_precision10 = mean_avg_precision10 / len(results)
    mean_avg_precision20 = mean_avg_precision20 / len(results)
    mean_avg_precision50 = mean_avg_precision50 / len(results)
    print('MEAN AVG PRECISION')
    print(mean_avg_precision10)
    print(mean_avg_precision20)
    print(mean_avg_precision50)
    print('\n')
    return results

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
    return relevance

def load_term_idf_weights():
    term_document_weights = {}
    document_terms = {}
    idf_list = {}
    with open(INDEXER_OUTPUT_FILE) as f_in:
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


def load_bmc():
    term_document_weights = {}
    document_terms = {}
    idf_list = {}
    with open(BMC_OUTPUT_FILE) as f_in:
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
    
def dump_term_idf_weights(term_document_weights, idf_list):
    with open(INDEXER_OUTPUT_FILE, "w") as write_file:
        for (token,idf) in idf_list.items():
            s = '%s:%.15f' % (token,idf)
            for docID in term_document_weights[token]:
                s += ';%s:%.15f' % (docID,term_document_weights[token][docID])
            write_file.write("%s\n" % s)

def dump_bmc(term_document_weights, idf_list):
    with open(BMC_OUTPUT_FILE, "w") as write_file:
        for (token,idf) in idf_list.items():
            s = '%s:%.15f' % (token,idf)
            for docID in term_document_weights[token]:
                s += ';%s:%.15f' % (docID,term_document_weights[token][docID])
            write_file.write("%s\n" % s)