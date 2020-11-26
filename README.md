Data Preprocess:
Kaggle dataset is preprocessed to get data files containing reviews pertaining to specific product.
A sample of this has been included in the food-reviews-sample-data folder. 

To execute Lsa :
Required Libraries:
1. Python3
2. Numpy 
3. NLTK library 
4. Apache Spark
5. Stopwords library from NLTK library 
To download this library run this command in terminal --->  python3 -m nltk.downloader all-corpora

For Extraction of Summary Sentences  
1. The file lsa.py imports lsa_extraction.py, lsa_read.py, tfidf_calculations.py 
2. Make sure all the above files arena the same folder when executing lsa.py
3. Execution format:
Spark-submit lsa.py -sentences <path to the input file> <path to the output directory/>
Example - spark-submit -sentences /Users/divyarshakoduri/Desktop/Document_Summarizer/LSA/B000NA8CWK.txt /Users/divyarshakoduri/Desktop/Document_Summarizer/LSA/output_lsa_sentences/

For Abstraction of summary keywords
1. The file lsa.py imports lsa_extraction.py, lsa_read.py, tfidf_calculations.py 
2. Make sure all the above files arena the same folder when executing lsa.py
3. Execution format:
Spark-submit lsa.py -words <path to the input file> <path to the output directory/>
Example - spark-submit -words /Users/divyarshakoduri/Desktop/Document_Summarizer/LSA/B000NA8CWK.txt /Users/divyarshakoduri/Desktop/Document_Summarizer/LSA/output_lsa_words/


To execute TextRank : 
Required Libraries:
1. Python3
2. Numpy 
3. NLTK library 
4. Apache Spark
5. Stopwords library from NLTK library 
To download this library run this command in terminal --->  python3 -m nltk.downloader all-corpora

1. The file textrank.py imports from textrank_helper.py
2. Make sure that both the files are in the same folder when executing textrank.py
3. Execution format
Spark-submit textrank.py <number of iterations> <number of sentences to be displayed in output> <path to the input file> <path to the output directory/>
Example - spark-submit 10 10 /Users/divyarshakoduri/Desktop/Document_Summarizer/LSA/B000NA8CWK.txt /Users/divyarshakoduri/Desktop/Document_Summarizer/LSA/output_textrank/

To execute using AWS EC2 
1. Go to EC2 from the AWS console and create an EC2 instance.
2. Upload all the required files to GitHub using the following commands
Clone your GitHub repo into your local terminal
Git add * - to add all the files in the directory
Git status - to check the status of all the files 
Git commit -m “” - to commit all the files into the github repositary 
Git push origin master - to push to the repository 
3. After pushing all the files to GitHub, open terminal of AWS EC2 and pull the git files.
4. Install all the required libraries(pyspark, nltk, dumpy,stopwords,all-corpora
5. Execute the desired program(lsa.py or textrank.py) on AWS EC2 terminal using the command:
python lsa.py -sentences B000NA8CWK.txt output_lsa/ 
python textrank.py 10 10 B000NA8CWK.txt output_lsa/
6.The output of the files can be visualized on the EC2 browser using a public IP address if configured appropriately.

