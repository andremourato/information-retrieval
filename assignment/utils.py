
import Stemmer
import json

def remove_non_alpha(string):
    return ''.join([ c if c.isalpha() else ' ' for c in string.lower()]).split() 

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

def dump_to_file(dic,filename):
    print('DUMPING TO FILE %s' % filename)
    with open("debug/%s" % filename, "w") as write_file:
        json.dump(dic, write_file, indent=4)
    