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

import re 
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    preprocess_text = ' '.join(stemmed_tokens)
    preprocess_text = re.sub(r'http\S+|www\S+', '', preprocess_text)
    preprocess_text = re.sub(r'\d+', '', preprocess_text)
    return preprocess_text

email= "Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize."
processed_email = preprocess_text(email)
print("Processed Email:", processed_email)

import pandas as pd
df=pd.read_csv('emails.csv', encoding='latin-1')
df.head()
df['processed_email'] = df['Message'].apply(preprocess_text)
print(df[['Message', 'processed_email']].head())
df_spam = df[df['Spam'] == 1]

import matplotlib.pyplot as plt
from wordcloud import WordCloud
spam_text = ' '.join(df_spam['processed_email'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Spam Emails')
plt.show()



