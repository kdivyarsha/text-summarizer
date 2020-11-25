import sys
from pyspark import SparkContext
import numpy
import tfidf_calculations
import lsa_read
import lsa_extraction

"""
Class implementing the Latent Semantic Analysis Algorithm
"""

class lsa_summarization:
	"""
	Function to extract summary sentences
	Here we calculate SVD of tfidf matrix
	choose k concepts from the right singular matrix and
	in each concept, select the sentence with largest value to the summary
	"""
	@staticmethod
	def sentance_summary(all_reviews, result_matrix, header_column):
		result = []
		U, S, VT = numpy.linalg.svd(result_matrix, full_matrices=0)
		concepts = lsa_extraction.sentence_extraction(VT,all_reviews,header_column)
		for i,concept in enumerate(concepts):
			for j,sent in enumerate(concept):
				result.append('[Concept '+str(i+1)+'][Sentence '+str(j+1)+'] :\t'+str(sent))
				print('[Concept '+str(i+1)+'][Sentence '+str(j+1)+'] :\t'+str(sent))
			print('\n')
		return result

	"""
	Function to abstract summary keywords
	Here we calculate SVD of tfidf matrix
	choose k concepts from the right singular matrix and
	in each concept, select the keywords with largest value to the summary
	"""
	@staticmethod
	def keywords_summary(all_reviews, result_matrix, header_row):
		result = []
		U, S, VT = numpy.linalg.svd(result_matrix.T, full_matrices=0)
		concepts = lsa_extraction.keywords_extraction(VT, header_row)
		for i,concept in enumerate(concepts):
			result.append('[Concept '+str(i+1)+'] :\t'+str(concept))
			print('[Concept '+str(i+1)+'] :\t'+str(concept))
		return result

if __name__=="__main__":
    if len(sys.argv) !=4:
        print("Usage :", sys.argv[0], "flag text_file output_file")
        print("Argument 1 : flag can take following values:")
        print(" -sentences: sentence summarization")
        print(" -words: keywords summarization")
        print("Argument 2 : text file to be provided")
        print("Argument 3 : output file to be provided")
        sys.exit()
    else:
        # defining a spark context used for establishing connection to spark cluster
	    sc = SparkContext(appName= 'Text_Summarization_using_LSA')
	    flag = sys.argv[1]
	    text_file = sys.argv[2]
	    output_file = sys.argv[3]

	    all_documents = sc.textFile(text_file).flatMap(lambda each_review: lsa_read.document_read(each_review))
	    all_reviews = sc.textFile(text_file).flatMap(lambda each_review: lsa_read.review_read(each_review))

	    term_frequency = all_documents.map(lambda k_wordlist: tfidf_calculations.term_frequency(k_wordlist[0], k_wordlist[1]))
	    vocabulary = tfidf_calculations.get_vocabulary(term_frequency)
	    document_freq_vector = tfidf_calculations.get_docfreq_vector(term_frequency,vocabulary )
	    term_frequency_matrix, header_column, header_row = tfidf_calculations.create_termfreq_matrix(term_frequency,vocabulary)
	    result_matrix = tfidf_calculations.create_result_matrix(term_frequency_matrix, document_freq_vector, header_column)
	    all_reviews.cache()

	    if flag == '-sentences':
	        output = lsa_summarization.sentance_summary(all_reviews, result_matrix, header_column)

	    elif flag == '-words':
	        output = lsa_summarization.keywords_summary(all_reviews, result_matrix, header_row)
	    sc.parallelize(output).coalesce(1).saveAsTextFile(sys.argv[3])


























