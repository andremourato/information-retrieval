import sys

from utils import *

def ltc_calculation(term_document_weights,document_term_weights,idf_list,queries):
    query_document_weights = {}
    document_query_weights = {}
    scores = {}
    for idx,query in enumerate(queries):
        for token in query:
            # If the token exists in at least one document
            for docID in document_term_weights:
                # Query document
                if token not in query_document_weights:
                    query_document_weights[token] = {}
                # Document Query
                if docID not in document_query_weights:
                    document_query_weights[docID] = {}

                if token in document_term_weights[docID]:
                    query_document_weights[token][docID] = idf_list[token] * term_document_weights[token][docID]
                    document_query_weights[docID][token] = idf_list[token] * term_document_weights[token][docID]
                else:
                    query_document_weights[token][docID] = 0
                    document_query_weights[docID][token] = 0
        # 3 - Score calculation
        scores[idx] = { docID: sum(v.values()) for docID,v in document_query_weights.items() }
        scores[idx] = dict(sorted(scores[idx].items(), key=operator.itemgetter(1), reverse=True))

    return query_document_weights, document_query_weights, scores


# def weighting_tf_idf(term_index,document_length_index,queries):

#     # 1 - LTC calculation
#     query_weights, document_query_weights, scores = ltc_calculation(term_document_weights,document_term_weights,idf_list,queries)
    
#     return document_term_weights, term_document_weights, query_weights, document_query_weights, scores, idf_list

# def weighting_bm25(term_index,document_length_index):
#     weights = {}

#     for docID in term_index:
#         for token in term_index[docID]:
#             if docID not in weights:
#                 weights[docID] = {}
#             weights[docID][token] = 1 + math.log10(term_index[docID][token])

#         norm_factor = 1/math.sqrt(sum([weights[docID][_]**2 for _ in weights[docID]]))

#         for token in weights[docID]:
#             weights[docID][token] *= norm_factor
#     return weights

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'outputs/indexer_output.txt'
    else:
        filename = sys.argv[1]
    print('Loading weights from',filename)
    print('------------------------------')

    term_document_weights, idf_list = load_term_idf_weights()

    dump_to_file(term_document_weights,'ranked_term_document_weights.json')

    dump_to_file(idf_list,'ranked_idf_list.json')

    # weights = weighting_bm25(term_index,document_length_index,queries)
    #document_term_weights, term_document_weights,query_weights, document_query_weights, scores, idf_list = weighting_tf_idf(term_index,document_length_index,queries)
