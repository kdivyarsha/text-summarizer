import re
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


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


def vertex_change(each_sent):
    review_id, sentence = each_sent[0], each_sent[1]
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    bag_of_words = re.findall(r'[a-zA-Z]+', sentence)
    bag_of_words = [lemmatizer.lemmatize(wrd.lower()) for wrd in bag_of_words if wrd.lower() not in stop_words]
    bag_of_words = [wrd for wrd in bag_of_words if len(wrd) > 3]
    return review_id, bag_of_words


def adjacencylist_creation(vert, total_vertices):
    sentence,vertex = vert[0],vert[1]
    edge_dictionary = {}
    for each_vert in total_vertices:
        each_edge = similarity_measure(vert, each_vert)
        if each_edge is not None:
            edge_dictionary[each_edge[0]] = each_edge[1]
    return (sentence, edge_dictionary)


def similarity_measure(ip1, ip2):
    sent1,vertex1 = ip1[0],ip1[1]
    sent2,vertex2 = ip2[0],ip2[1]
    if sent1 != sent2: #To skip measuring between same sentence
        common_words = len(set(vertex1).intersection(vertex2))
        log_of_len  = np.log2(len(vertex1)) + np.log2(len(vertex2))
        similar_value = common_words/(log_of_len + 1)
        if similar_value != 0:
            return (sent2, similar_value)
