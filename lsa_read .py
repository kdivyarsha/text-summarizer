import numpy
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re

"""
Function to read input file and divide into lines.
Each line is further broken down into list of words
removes sentences which have less than 6 words and words having less than 4 letters
input : line
output : list of words
"""
def document_read(line):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    each_review = line.split("\t")
    each_review_id = each_review[0]
    sentences = each_review[5].split(".")
    output = []
    for index, sentence in enumerate(sentences):
        sentence_id = each_review_id + '_' + str(index)
        sentence_len = len(sentence.split(" "))
        if 10 < sentence_len < 30:
            words = re.findall(r'[a-zA-Z]+', sentence)
            words = [lemmatizer.lemmatize(w.lower()) for w in words if w.lower() not in stop_words]
            words = [w for w in words if len(w) > 3]
            output.append((sentence_id, words))
    return output

"""
Function reads each line and divides itno reviewid and review
input : line
output : reviewid - review formatted output
"""
def review_read(line):
    each_review = line.split("\t")
    each_review_id = each_review[0]
    sentences = each_review[5].split(".")
    output = []
    for index, sentence in enumerate(sentences):
        sentence_id = each_review_id + '_' + str(index)
        output.append((sentence_id, sentence))
    return output