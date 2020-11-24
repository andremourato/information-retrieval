
import operator

a = [1,2,3]
b = {'a':1,'c':34,'b':0}

print(list(b.keys()))

for i, item in enumerate(list(b.keys())):
    print(i,item)
