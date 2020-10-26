import Stemmer
print(Stemmer.algorithms())

stemmer = Stemmer.Stemmer('porter')

print(stemmer.stemWord('cycling'))

print(stemmer.stemWords(['cycling', 'cyclist']))

print(stemmer.stemWords(['cycling', b'cyclist']))

print(stemmer.maxCacheSize)

stemmer.maxCacheSize = 1000

print(stemmer.maxCacheSize)