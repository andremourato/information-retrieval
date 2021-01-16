###############################
#   Authors             
###############################
#   André Mourato nmec 84745
#   Gonçalo Marques nmec 80327
###############################
# Benchmarking
import tracemalloc
import time

# Necessary imports
import sys

#file imports
from Indexer import *

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'datasets/metadata_small.csv'
    else:
        filename = sys.argv[1]
    print('Reading dataset from file',filename)
    print('------------------------------------------------------------')
    print('STARTING INDEXING...')
    print('------------------------------------------------------------')

    ind = Indexer(filename,2)
    # Trace used memory and time for the 
    # indexing process using the improved tokenizer
    tracemalloc.start()
    time_start = time.process_time()

    #########################################################
    # INDEXER
    #########################################################
    # 1 - Indexing
    ind.indexer()
    print('Total vocabulary size is: ',ind.num_tokens,'words')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage when indexing was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    
    # # 2 - TF-IDF
    # ind.lnc_calculation()
    # Indexer.dump_weights(ind.term_document_weights, ind.idf_list, 'tf_idf_weights.csv')
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Memory usage when calculating lnc was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    
    # # # 3 - BMC
    # ind.bmc_pre_calculation(ind.document_length_index, ind.idf_list)
    # dump_weights(ind.bmc_weights, ind.idf_list, 'bmc_weights.csv')
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Memory usage when calculating bmc was {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # #########################################################
    # # BENCHMARKING INFORMATION
    # #########################################################
    print('Total indexing time:',time.process_time() - time_start,'s')
    current, peak = tracemalloc.get_traced_memory()
    print(f"FINAL MEMORY USAGE: {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print('------------------------------------------------------------')
    