import csv
import re

def load_stop_words(file):
    with open(file)  as f_in:
        return list(map(lambda x: x.split()[0],f_in.readlines()))

def read_corpus(file):
    with open(file)  as csvfile:
        reader = csv.DictReader(csvfile)
        documentList = []
        for row in reader:
            if len(row['abstract']) > 0:
                documentList.append(row['title'] + row['abstract'])

    return documentList

def remove_chars(string):
    char_list = ['(',')','&ndash;','.',',','%','[',']',':','/','\'','\\','-']
    for c in char_list:
        string = string.replace(c,' ')
    return string

def simple_tokenizer(lst):
    documentList = []
    for document in  lst:
        tokenlist = []
        document = remove_chars(document)
        for token in document.lower().split():
            if len(token) >= 3:
                tokenlist.append(token)
        documentList.append(tokenlist)
    return documentList

#def improved_tokenizer(lst):


def indexer(lst):
    count_index = {}
    document_index = {}
    for docID in range(len(lst)):
        for token in lst[docID]:
            if token not in count_index.keys():
                count_index[token] = 1
            else:
                count_index[token] += 1
            if token not in document_index.keys():
                document_index[token] = {}
            document_index[token][docID] = True
    return count_index, document_index

stopwords = load_stop_words('stopwords.txt')

file = 'all_sources_metadata_2020-03-13.csv'
lst = read_corpus(file)
index = simple_tokenizer(lst[:2])
count_index, document_index = indexer(index)
# print(count_index)
print(document_index)
