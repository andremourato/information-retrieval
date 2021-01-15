from utils import *
import os
import math

class Spimi:

    folder = 'debug/'

    @staticmethod
    def sort_terms(terms):
        res = { k:terms[k] for k in sorted(terms.keys())}
        terms.clear()
        return res

        
    #term:idf:docid_doc_weight:po1,pos2,po3:docid:doc_weight:pos1....   
    @staticmethod
    def merge_blocks(N,T):
        
        num_sorted = 0
        max_num_terms_per_block = math.floor(T/N)
        indexes = [0] * N
        result = {}
        #merges
        completed = set()
        while len(completed) < N:
            lowest_term = '{'
            lowest_indexes = [0]
            #reads the lowest term from each block
            for i in range(N):
                if i not in completed:
                    with open(os.path.join(Spimi.folder,'block_%d.json'%(i+1))) as block:
                        data = json.load(block)
                        lowest_in_block = list(data.keys())[indexes[i]]
                        if lowest_term > lowest_in_block:
                            lowest_term = lowest_in_block
                            lowest_indexes = [i]
                        elif lowest_term == lowest_in_block:
                            lowest_indexes.append(i)

            for l in lowest_indexes:
                with open(os.path.join(Spimi.folder,'block_%d.json'%(l+1))) as block:
                    data = json.load(block)
                    #updates block indexes
                    indexes[l] += 1
                    if indexes[l] == len(data):
                        completed.add(l)
                    if lowest_term in result:
                        result[lowest_term].update(data[lowest_term])
                    else:
                        result[lowest_term] = data[lowest_term]

            #check if result is full
            if len(result) >= max_num_terms_per_block or len(completed) == N:
                num_sorted += 1
                dump_to_file(result,'sorted_block_%d.json'%num_sorted)
                result.clear()
                result = {}
