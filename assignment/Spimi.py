from utils import *
import os
import math

class Spimi:

    FOLDER = 'debug/'

    @staticmethod
    def sort_terms(terms):
        res = { k:terms[k] for k in sorted(terms.keys())}
        terms.clear()
        return res
        
    #term:idf:docid_doc_weight:po1,pos2,po3:docid:doc_weight:pos1....   
    @staticmethod
    def merge_blocks(N,T):
        NUM_LOADED = 0
        MAX_INT = 9223372036854775807
        num_sorted = 0
        max_num_terms_per_block = math.floor(T/N)
        indexes = [0] * N
        sizes = [MAX_INT] * N
        result = {}
        #merges
        num_completed = 0
        while num_completed < N:
            lowest_term = '{'
            lowest_indexes = [0]
            #reads the lowest term from each block
            for i in range(N):
                if indexes[i] < sizes[i]:
                    NUM_LOADED += 1
                    with open(os.path.join(Spimi.FOLDER,'block_%d.json'%(i+1))) as block:
                        data = json.load(block)
                        sizes[i] = len(data)
                        lowest_in_block = list(data.keys())[indexes[i]]
                        if lowest_term > lowest_in_block:
                            lowest_term = lowest_in_block
                            lowest_indexes = [i]
                        elif lowest_term == lowest_in_block:
                            lowest_indexes.append(i)
                        data.clear()

            for l in lowest_indexes:
                with open(os.path.join(Spimi.FOLDER,'block_%d.json'%(l+1))) as block:
                    data = json.load(block)
                    #updates block indexes
                    indexes[l] += 1
                    if indexes[l] == sizes[l]:
                        num_completed += 1
                    if lowest_term in result:
                        result[lowest_term].update(data[lowest_term])
                    else:
                        result[lowest_term] = data[lowest_term]
                    data.clear()

            #check if result is full
            if len(result) >= max_num_terms_per_block or num_completed == N:
                num_sorted += 1
                dump_to_file(result,'sorted_block_%d.json'%num_sorted)
                result.clear()
                result = {}
