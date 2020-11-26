import re
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

"""
Function that reads one line and splits into a tuple containing reviewid and review
input : line
output : tuple containing reviewid and review
"""
def create_graph_vert(sent):
    each_review = sent.split("\t")
    each_review_id = each_review[0]
    sentence_split = each_review[5].split(".")
    output = []
    for index, sentence in enumerate(sentence_split):
        sentence_id = each_review_id + '_' + str(index)
        sentence_len = len(sentence.split(" "))
        if 10 < sentence_len < 30:
            output.append((sentence_id, sentence))
    return output

"""
Function that takes tuple reviewid and review and creates wordlist for each review
input : tuple containing reviewid and review
output : tuple containg reviewid and wordlist
"""
def vertex_change(each_sent):
    review_id, sentence = each_sent[0], each_sent[1]
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    bag_of_words = re.findall(r'[a-zA-Z]+', sentence)
    bag_of_words = [lemmatizer.lemmatize(wrd.lower()) for wrd in bag_of_words if wrd.lower() not in stop_words]
    bag_of_words = [wrd for wrd in bag_of_words if len(wrd) > 3]
    return review_id, bag_of_words

"""
Function that takes a tuple reviewid and review and creates adjacency list for this tuple with other reviews
It measures similarity between reviews of tuples 
input : tuple with reviewid and review and all such tuples
output : reviewid and dictionary of other review id with corresponding similarity values
"""
def adjacencylist_creation(vert, total_vertices):
    sentence,vertex = vert[0],vert[1]
    edge_dictionary = {}
    for each_vert in total_vertices:
        each_edge = similarity_measure(vert, each_vert)
        if each_edge is not None:
            edge_dictionary[each_edge[0]] = each_edge[1]
    return (sentence, edge_dictionary)

"""
Function to measure similarity between two tuples of reviewid and words
here similarity is computed using the formula
    sim = (count of common words between tuples)/ (1+ log(len(tuple1)) + log(len(tuple2))) 
"""
def similarity_measure(ip1, ip2):
    sent1,vertex1 = ip1[0],ip1[1]
    sent2,vertex2 = ip2[0],ip2[1]
    if sent1 != sent2: #To skip measuring between same sentence
        common_words = len(set(vertex1).intersection(vertex2))
        log_of_len  = np.log2(len(vertex1)) + np.log2(len(vertex2))
        similar_value = common_words/(log_of_len + 1)
        if similar_value != 0:
            return (sent2, similar_value)
