import numpy
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re

""" 
Function to calculate term frequency 
input : vocabulary
output : term frequency matrix
"""
def get_tf_matrix(vocabulary, tfdict):
    return [tfdict.get(word, 0) for word in vocabulary]

""" 
Function to calculate document frequency 
input : vocabulary
output : document frequency matrix
"""
def get_df_matrix(vocabulary, tfdict):
    return [ 1.0 if (tfdict.get(word, 0) > 0) else 0. for word in vocabulary]

""" 
Function takes the review id and wordlist and calculates the term frequency
input : review id and word list
output : term frequency corresponding to each review id
"""
def term_frequency(review_id, list_words):
    termfrequency = dict()
    for term in list_words:
        termfrequency[term] = termfrequency.get(term, 0.0) + 1.0
    return review_id, termfrequency

"""
Function to compute Inverse Document Frequency using the document-frequency vector and total number of sentences
input : document-frequency vector and total number of sentences
output : Inverse Document Frequency
"""
def inverse_doc_frequecy(num, document_freq):
    return numpy.log10(numpy.reciprocal(document_freq) * num)

"""
Function to accumulate full vocabulary from the given input
input : term frequency
output : vocabulary
"""
def get_vocabulary(term_frequency):
	vocabulary = term_frequency.map(lambda tuple: list(tuple[1].keys())).reduce(lambda x,y: x + y)
	vocabulary = numpy.unique(vocabulary)
	return vocabulary

"""
Function to compute document frequency vector
Here document means sentences in the review file
input : term frequency matrix and vocabulary
output : document frequency vector
"""
def get_docfreq_vector(term_frequency,vocabulary):
	documentfvector = term_frequency.map(lambda tuple: get_df_matrix(vocabulary,tuple[1])).reduce(lambda x, y: numpy.array(x) + numpy.array(y))
	return documentfvector

"""
Function to compute term-frequency vector for each selected sentence
input : term frequency matrix and vocabulary
output : term frequency matrix, column header and row header
"""
def create_termfreq_matrix(term_frequency,vocabulary):
	tf = term_frequency.map(lambda rev_id_tfdict: (rev_id_tfdict[0], get_tf_matrix(vocabulary, rev_id_tfdict[1]))).sortByKey()
	a = tf.collect()
	termfmatrix = [i[1] for i in a]
	header_col = [i[0] for i in a]
	header_row = vocabulary
	return termfmatrix, header_col, header_row

"""
Function to compute TFIDF matrix by multiplying TF matrix with IDF vector using numpy
Here the rows and columns of the TF-IDF matrix are words and sentences
input : term frequency matrix, document frequency vector and column header
output : tf idf matrix
"""
def create_result_matrix(term_frequency_matrix, document_freq_vector, header_column):
	term_frequency_matrix = numpy.array(numpy.transpose(term_frequency_matrix))
	document_freq_vector = inverse_doc_frequecy(len(header_column), document_freq_vector)
	document_freq_vector = numpy.array(numpy.transpose(document_freq_vector))
	document_freq_vector = numpy.reshape(document_freq_vector, (-1,1))
	result_matrix = term_frequency_matrix * document_freq_vector
	return result_matrix