import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
text = "Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize."
tokens = word_tokenize(text.lower())
tokens = [word for word in tokens if word not in string.punctuation]

print("Tokens:", tokens)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]
print("Filtered Tokens:", filtered_tokens)
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)
