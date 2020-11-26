import sys
import textrank_helper
from pyspark import SparkContext


class graph:
    @staticmethod
    def create_vertices(review_sentences):
        # Create vertices of graph that contains review_id and wordlist from review sentences
        graph_vertices = review_sentences.map(lambda l: textrank_helper.vertex_change(l))
        return graph_vertices

    @staticmethod
    def create_adjacency_list(graph_vertices):
        #Construct graph by creating adjacency list for each vertex
        total_vertices = graph_vertices.collect()
        final_graph = graph_vertices.map(lambda vertex: textrank_helper.adjacencylist_creation(vertex, total_vertices))
        return final_graph

    @staticmethod
    def filter_vertices(final_graph):
        # Filter only vertices that have atleast one node in its adj list.
        # Remove this filter step to save time if the graph is highly connected
        filter_graph = final_graph.filter(lambda node: len(node[1]) > 0)
        filtered_graph = filter_graph.cache()
        return filtered_graph


class rank:
    @staticmethod
    def set_ranks(filtered_graph):
        # Set initial rank to all sentences as 0.15
        set_rank = filtered_graph.map(lambda neighbour_vertices: (neighbour_vertices[0], 0.15))
        return set_rank

    @staticmethod
    def calculate_ranks(filtered_graph,num_iterations,set_rank):
            # Iterative TextRank algorithm which will converge after many iterations
        for i in range(0,num_iterations):
            # Compute contribution by each vertex to its all neighbours and sum the contribution to get the updated rank
            # TextRank Formula: TR(Vi) = (1-d) + d* SUM,Vj in In(Vi) [[Wji * PR(Vj)]/ [SUM, Vk in Out(Vj) [Wjk]]]
            neighbour_contribution = filtered_graph.join(set_rank).flatMap(lambda sentence_neighbour_dictionary: contribute.neighbour_contribution(sentence_neighbour_dictionary[1][0], sentence_neighbour_dictionary[1][1]))
            set_rank = neighbour_contribution.reduceByKey(lambda x,y: x+y).mapValues(lambda each_rank: 0.15 + 0.85 * each_rank)
        return set_rank

    @staticmethod
    def sort_ranks(rank_calculation, sentence_count, review_sentences):
        output = []
        # Print the sentences that have higher rank
        output_rank = rank_calculation.collect()
        result_rank = sorted(output_rank, key=lambda x: x[1], reverse=True)

        for j in range(0, sentence_count ):
            output.append('Rank: ' + str(round(result_rank[j][1],2)) + '\t\tSentence : ' + str(review_sentences.lookup(result_rank[j][0])))
            print('Rank: ' + str(round(result_rank[j][1],2)) + '\t\tSentence : ' + str(review_sentences.lookup(result_rank[j][0])))
        return output


class contribute:
    @staticmethod
    def neighbour_contribution(neigh_dict, rank):
        """
        Compute the contribution from a node using it's current rank and its neighbours' weight
        :param neigh_dict: A dictionary which contains {neighbour_nodes : weights(computed using similarity)}
        :param rank: current rank of the node
        :return:list of (node, contribution received) from the parent node for which we runnig this method
        """
        result = []
        outweight = sum(neigh_dict.values())
        for key,weight in neigh_dict.items():
            contrib = (rank*weight)/outweight
            result.append((key, contrib))
        return result

if __name__=="__main__": 
    sc = SparkContext(appName = 'TextRank')#,master = 'local',  pyFiles = ['trhelp.py', 'dataprep.py'])
    path = "/Users/thejashreekunam/Documents/attachments/B0033YXSHW.txt"
    num_iterations = 10
    sentence_count = 10

    # Get sentences from the fine food reviews with a distinct id for each one 
    review_sentences = sc.textFile(path).flatMap(lambda review: textrank_helper.create_graph_vert(review))

    graph_vertices = graph.create_vertices(review_sentences)
    review_sentences = review_sentences.cache()

    final_graph = graph.create_adjacency_list(graph_vertices)

    filtered_graph = graph.filter_vertices(final_graph)

    set_rank = rank.set_ranks(filtered_graph)

    rank_calculation = rank.calculate_ranks(filtered_graph,num_iterations,set_rank)

    sorted_ranks = rank.sort_ranks(rank_calculation, sentence_count, review_sentences) 

    sc.parallelize(sorted_ranks).coalesce(1).saveAsTextFile("output-textrankfinal1/")

