document_index = { "a" : 100, "b" : 4, "c" : 10 }

print(sorted(document_index, key = lambda key: document_index[key]))