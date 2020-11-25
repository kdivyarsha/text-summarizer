import numpy
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re

"""
Function takes right singular matrix of SVD and returns summary of the matrix
"""
def sentence_extraction(VT, reviews, columnheader, num_concept=10, num=5):
    concepts_output = []
    # for idxs in numpy.argpartition(VT[:k,:], -n, 1)[:,-n:]:
    for indexs in numpy.fliplr(VT[:num_concept,:].argsort()[:,-num:]):
        main_sentences = []
        for index in indexs:
            main_sentences.append(reviews.lookup(columnheader[index]))
        concepts_output.append(main_sentences)
    return concepts_output

"""
Function that takes right singular matrix of SVD and returns summary of keywords
"""
def keywords_extraction(VT, rowheader, num_concept = 10, num = 5):
    concepts_output = []
    for indexs in numpy.fliplr(VT[:num_concept,:].argsort()[:,-num:]):
        keywords = []
        for index in indexs:
            keywords.append(rowheader[index])
        concepts_output.append(keywords)
    return concepts_output