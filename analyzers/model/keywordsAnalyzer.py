import pandas as pd
import string
import nltk
import spacy
import yake
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words("english"))
# python -m spacy download en_core_web_sm
#nlp = spacy.load("en_core_web_sm")

class KeywordsAnalyzer:
    def __init__(self, top_n=5):
        self.kw_extractor = yake.KeywordExtractor(top=top_n, stopwords=stop_words)

    def extract_keywords(self, text):
        keywords = self.kw_extractor.extract_keywords(text)
        return [kw for kw, _ in keywords]

if __name__ == "__main__":

    # ========== 1. Generate simulated news data ==========
    sample_news = [
        "The stock market surged today after the Federal Reserve announced it would pause interest rate hikes.",
        "Scientists have discovered a new species of frog in the Amazon rainforest, highlighting the region's biodiversity.",
        "Apple unveiled its latest iPhone with new AI-powered features and improved battery life.",
        "The global climate summit concluded with a landmark agreement on carbon emissions reduction by 2030.",
        "A powerful earthquake struck Japan's northeast coast, causing damage but no major injuries reported."
    ]

    df = pd.DataFrame({'text': sample_news})
    analyzer = KeywordsAnalyzer(top_n=5)
    keywords = analyzer.extract_keywords(df['text'][0])
    print(keywords)

    exit()

    # ========== 2. Text preprocessing function ==========
    def preprocess(text):
        tokens = word_tokenize(text.lower())
        return " ".join([
            t for t in tokens if t.isalpha() and t not in stop_words
        ])

    df['clean_text'] = df['text'].apply(preprocess)

    # ========== 3. Use YAKE to extract keywords ==========
    analyzer = KeywordsAnalyzer(top_n=5)
    all_keywords = []
    for text in df['text']:
        keywords = analyzer.extract_keywords(text)
        all_keywords.append(keywords)
    print(all_keywords)
    # Show data
    print(df[['text', 'clean_text']])
    print(all_keywords)


