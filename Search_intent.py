import pandas as pd
import requests
import numpy as np
import json
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

apikey= "AIzaSyB7CpDy_L9yU5L9TQD5xOlyg2lPPt7DDiU"
file_path = r"C:\Users\Admin\Downloads\Top Queries.xlsx"
df = pd.read_excel(file_path) 
total_queries = len(df.index)
query_list = df['Top Queries'].tolist()
pd.Series(query_list).value_counts().head(10)

nltk.download('stopwords')
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

#extract bigrams/trigrams
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3)) 
# matrix of ngrams
ngrams = c_vec.fit_transform(df['Top Queries'])
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
df_ngram.head(10)

#new code
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Define the stoplist - replace with your actual list of stopwords
stoplist = ["your", "list", "of", "stopwords"]

# Corrected file path handling
file_path = r"C:\Users\Admin\Downloads\Top Queries.xlsx"

# Reading the Excel file
try:
    df = pd.read_excel(file_path)
    print("Excel file read successfully.")
except FileNotFoundError:
    print("The specified file was not found.")
    df = None
except Exception as e:
    print(f"An error occurred: {e}")
    df = None

if df is not None:
    # Print the DataFrame to inspect it
    print("DataFrame loaded from the Excel file:")
    print(df)

    # Check if 'Top Queries' column exists
    if 'Top Queries' in df.columns:
        # Ensure all entries in the 'Top Queries' column are converted to strings and then to lowercase
        df['Top Queries'] = df['Top Queries'].astype(str).str.lower()

        # Create CountVectorizer for bigrams/trigrams
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2, 3))
        
        # Matrix of n-grams
        ngrams = c_vec.fit_transform(df['Top Queries'])
        
        # Count frequency of n-grams
        count_values = ngrams.toarray().sum(axis=0)
        
        # List of n-grams
        vocab = c_vec.vocabulary_
        
        # Create DataFrame of n-gram frequencies
        df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
                               ).rename(columns={0: 'frequency', 1: 'bigram/trigram'})
        
        # Display the top 10 n-grams
        print(df_ngram.head(10))
    else:
        print("'Top Queries' column not found in the DataFrame.")
