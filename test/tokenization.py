import sys
import re

print('file',sys.argv[1])
occur = {}
stop = []

with open('stop_words.txt') as fstop:
    stop = [word.rstrip() for word in fstop.readlines()]

with open(sys.argv[1]) as f:
    for line in f.readlines():
        line = line.replace('\n',' ')
        line = line.replace("--",' ')
        line = line.replace("'",' ')
        line = re.sub(r"[,.;@#?!&$]+\ *", " ", line)
        for word in line.split(' '):
            if len(word) < 4:
                continue
            if word in stop:
                continue
            if word in occur.keys():
                occur[word] += 1
            else:
                occur[word] = 1

with open(sys.argv[1]+'_output.txt','w') as fout:
    sort_orders = sorted(occur.items(), key=lambda x: x[1], reverse=True)
    fout.write(str(sort_orders))