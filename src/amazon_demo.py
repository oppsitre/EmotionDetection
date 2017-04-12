from emotionLDA import *
from preprocess import Comment
import os
import urllib
import tarfile
vocabSize = 50000

# cmt = Comment('4704894', '../dataset/')
#
# reviews = cmt.content
reviews = []
with open('texts.txt' , 'r') as f:
    for line in f:
        line = line.strip()
        reviews.append(line)

sampler = SentimentLDAGibbsSampler(5, 2.5, 0.1, 0.3)
sampler.run(reviews, 2000, None, True)
#
# sampler.getTopKWords(25)
sampler.get_wordmap()
sampler.get_tassign()
