###############################
#   Authors             
###############################
#   André Mourato nmec 84745
#   Gonçalo Marques nmec 80327
###############################
# Necessary imports
import math
import Stemmer
import csv
# File imports
from utils import *
from Spimi import *

class Indexer:

    @staticmethod
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

    def __init__(self,filename,block_size_limit):
        self.BLOCK_SIZE_LIMIT = block_size_limit #max number of documents per block 
        self.DATASET_FILENAME = filename
        self.stopwords = Indexer.load_stop_words('resources/stopwords.txt')
        self.num_blocks = 0
        self.num_tokens = 0
        self.document_length_index = {}
        #LNC CALCULATION
        self.idf_list = {}


    def indexer(self):
        '''An improved tokenizer that replaces all non-alphabetic characters by a space, lowercases
        tokens, splits on whitespace, and ignores all tokens with less than 3 characters. This tokenizer
        also uses the Porter stemmer and applies a stopword filter
        ----------
        filename : string
            File containing the dataset
            
        Returns
        -------
        term_index : dict
            Dictionary that contains the token as the key and the number of occurences as the value.
            Example: {
                "9dj07sac": {
                    "incub": 4,
                    "period": 4,
                    "epidemiolog": 2,
                    "characterist": 1,
                    "novel": 2
                }
            }

        document_length_index : dict
            Dictionary that contains the number of terms of each document, i.e., the document length
            Example: {
                "vho70jcx": 124,
                "i9tbix2v": 92,
                "62gfisc6": 147,
                "058r9486": 153,
                "wich35l7": 161,
                "z3tgnzth": 88,
                "1xxrnpg3": 127,
            }
        '''
        term_index = {}
        doc_number = 0
        with open(self.DATASET_FILENAME) as csvfile:
            # Iterate over the CSV file ignoring entries without an abstract
            # and joining the title and abstract fields into a single string
            
            for idx,row in enumerate(csv.DictReader(csvfile)):
                if len(row['abstract']) > 0:
                    string =  row['title'] + ' ' + row['abstract']
                    # Removes non-alphabetic characters by a space, lowercases
                    # tokens, splits on whitespace, and ignores all tokens with less than 3 characters.
                    # This tokenizer also uses the Porter stemmer and applies a stopword filter
                    token_list = Stemmer.Stemmer('porter').stemWords([token \
                                    for token in (remove_non_alpha(string)) \
                                        if len(token) >= 3 and token not in self.stopwords])
                    for tok_idx in range(len(token_list)):
                        tok = token_list[tok_idx]
                        self.num_tokens += 1 #counts tokens
                        # Indexes all the input tokens into one dictionaries
                        # the term_index dict which registers the total number of occurrences
                        # of a token in each document
                        # Counts the number of tokens in each document
                        if row['cord_uid'] not in self.document_length_index:
                            self.document_length_index[row['cord_uid']] = 0
                        # Counts the number of terms in each document
                        self.document_length_index[row['cord_uid']] += 1

                        # Counts the term frequency
                        # if row['cord_uid'] not in term_index:
                        #     term_index[row['cord_uid']] = {}
                        # if tok not in term_index[row['cord_uid']]:
                        #     term_index[row['cord_uid']][tok] = 1
                        # else:
                        #     term_index[row['cord_uid']][tok] += 1

                        # Counts the term frequency
                        if tok not in term_index:
                            term_index[tok] = {}
                        if row['cord_uid'] not in term_index[tok]:
                            term_index[tok][row['cord_uid']] = [tok_idx]
                        else:
                            term_index[tok][row['cord_uid']] += [tok_idx]

                    doc_number = (doc_number + 1) % self.BLOCK_SIZE_LIMIT
                    if doc_number == 0:
                        self.num_blocks += 1
                        dump_to_file(Spimi.sort_terms(term_index),'block_'+ str(self.num_blocks) +'.json')
            # writes the last block
            if len(term_index) > 0:
                self.num_blocks += 1
                dump_to_file(Spimi.sort_terms(term_index),'block_'+ str(self.num_blocks) +'.json')


        ### MERGE BLOCKS ###
        Spimi.merge_blocks(self.num_blocks,self.num_tokens)

    def lnc_calculation(self,N):
        '''Normalized lnc weight and idf calculator for all terms in dataset
        ----------
        N : int
            Number of documents in the dataset
            
        Returns
        -------
        term_document_weights : dict
            Dictionary of dictionaries that associates the lnc normalized weight of all
            documents that contain each term
            Example: {
                "siann": {
                    "vho70jcx": 0.1646582242160933
                },
                "strain": {
                    "vho70jcx": 0.1431574201623654,
                    "nljskxut": 0.1081438046556659,
                    "lcpp5fim": 0.17536757263696706,
                    "gjxumrmm": 0.1248067511230974
                    }
                }

        idf_list : dict
            Dictionary that contains the token as the key and the idf as the value.
            Example: {
                "siann": 4.578776695691345,
                "strain": 0.9240227624384145,
                "identif": 1.2125405719730515,
                "align": 2.012928877017827
                }
        '''
        term_document_weights = {}
        for docID in term_index:
            # 1 - NON-NORMALIZED WEIGHT CALCULATION
            document_term_weights = {}
            for token in term_index[docID]:
                document_term_weights[token] = 1 + math.log10(term_index[docID][token])
                if token not in term_document_weights:
                    term_document_weights[token] = {}
                term_document_weights[token][docID] = 1 + math.log10(term_index[docID][token])
        
            # 2 - CALCULATION OF THE NORM FACTOR
            norm_factor = 1/math.sqrt(sum([w**2 for w in document_term_weights.values()]))
            
            # 3 - NORMALIZED WEIGHT CALCULATION
            for token in document_term_weights:
                term_document_weights[token][docID] *= norm_factor
                # 4 - Calculating IDF
                N = len(self.document_length_index)
                dft = len(term_document_weights[token])
                self.idf_list[token] = math.log10(N/dft)

    def bm25_avdl(self):
        '''Calculates average document length of the dataset, used for bm25
        ----------
        document_length_index : dict
            Dictionary that contains the token as the key and the number of terms as the value.
            
        Returns
        -------
        avdl : float
            Average document length
        '''
        avdl = sum(v for v in self.document_length_index.values())
        return avdl / len(self.document_length_index)

    def bm25_weighting(self,N, k, b, avdl, document_term_index):
        '''Calculates bm25 weights for each token
        ----------
        N : int
            Total number of documents

        k : double
            Term frequency saturation value

        b : double
            Document length normalization factor

        avdl : float
            Average document length

        document_term_index : dict
            Dictionary that contains the token as the key and the number of occurences as the value.

        document_length_index : dict
            Dictionary that contains the token as the key and the number of terms as the value.

        idf_list : dict
            Dictionary that contains the token as the key and the idf as the value.
            
        Returns
        -------
            weights : dict
                Dictionary of dictionaries that contains the term as the key 
                and a dictionary with the docIDs in which the term exists and corresponding lnc normalized weight,
                as the value.
            Example :{
                "siann": {
                    "vho70jcx": 8.205306314501156
                },
                "strain": {
                    "vho70jcx": 1.4736887259523785,
                    "nljskxut": 1.050309414805419,
                    "lcpp5fim": 1.4952129090104211,
                    "gjxumrmm": 1.2983115687733342
                    }
                }
        '''
        for docID in document_term_index:
            for token in document_term_index[docID]:
                if token not in self.weights:
                    self.weights[token] = {}
                # Calculates the weight of term token in docID
                first = self.idf_list[token]
                second = (k+1) * document_term_index[docID][token]
                third = 1 / ( k*((1-b) + (b*self.document_length_index[docID] / avdl)) + document_term_index[docID][token])
                self.weights[token][docID] = first*second*third 
        return self.weights
                    

    def bmc_pre_calculation(term_index):
        '''Uses parameter values required for bm25 weighting and then calculates bm25 weights for each token
        ----------
        term_index : dict
            Dictionary that contains the token as the key and the number of occurences as the value.

        document_length_index : dict
            Dictionary that contains the token as the key and the number of terms as the value.

        idf_list : dict
            Dictionary that contains the token as the key and the idf as the value.
            
        Returns
        -------
            weights : dict
                Dictionary of dictionaries that contains the term as the key 
                and a dictionary with the docIDs in which the term exists and the corresponding lnc normalized weight,
                as the value.
            Example :{
                "siann": {
                    "vho70jcx": 8.205306314501156
                },
                "strain": {
                    "vho70jcx": 1.4736887259523785,
                    "nljskxut": 1.050309414805419,
                    "lcpp5fim": 1.4952129090104211,
                    "gjxumrmm": 1.2983115687733342
                    }
                }
        '''
        avdl = bm25_avdl(self.document_length_index)
        N = len(self.document_length_index)
        k = 1.2
        b = 0.75
        return bm25_weighting(N, k, b, avdl, term_index, self.document_length_index, self.idf_list)

    