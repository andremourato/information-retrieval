import Stemmer
import json
import os

INDEXER_OUTPUT_FILE = 'outputs/indexer_output.txt'
DEBUG_DIR = 'debug/'

#########################################################
# AUXILIAR METHODS
#########################################################
def remove_non_alpha(string):
    return ''.join([ c if c.isalpha() else ' ' for c in string.lower()]).split()

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