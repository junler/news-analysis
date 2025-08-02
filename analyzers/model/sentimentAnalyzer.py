import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# download necessary NLTK resources
nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class NewsSentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None

    # load multi-domain news sentiment analysis dataset
    def load_external_dataset(self, file_path='./data/sentiment/multi_domain_combined.csv'):
        try:
            # try different encodings to read data
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, names=['sentiment', 'text'], encoding=encoding)
                    print(f"successfully read multi-domain dataset using {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print("cannot read multi-domain dataset, use kaggle dataset instead...")
                # use kaggle dataset instead
                finance_path = '../../data/sentiment/all-data.csv'
                for encoding in encodings:
                    try:
                        df = pd.read_csv(finance_path, names=['sentiment', 'text'], encoding=encoding)
                        print(f"successfully read kaggle dataset using {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("cannot read any dataset, please ensure the data file exists and is formatted correctly")
            print("\ndata info: \n")
            print(df.info())
            print("\ndata null: \n")
            print(df.isnull().sum())
            print("\ndata describe: \n")
            print(df.describe())
            # clean data
            df = df.dropna()  # remove empty values
            df['text'] = df['text'].astype(str)  # ensure text is a string type
            
            # map sentiment labels to numeric values
            sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            df['sentiment_numeric'] = df['sentiment'].map(sentiment_map) # ignore the warning
            
            # remove labels that cannot be mapped
            df = df.dropna(subset=['sentiment_numeric'])
            df['sentiment_numeric'] = df['sentiment_numeric'].astype(int)
            
            # verify the validity of the dataset
            if len(df) == 0:
                raise ValueError("dataset is empty, please check the data file content")
            
            print(f"dataset size: {len(df)} news")
            print("sentiment label distribution:")
            sentiment_distribution = df['sentiment'].value_counts()
            print(sentiment_distribution)
            
            # check the quality of the dataset
            avg_length = df['text'].str.len().mean()
            print(f"average text length: {avg_length:.1f} characters")
            
            # show the balance of the dataset
            total = len(df)
            balance_info = {}
            for sentiment in sentiment_distribution.index:
                count = sentiment_distribution[sentiment]
                percentage = (count / total) * 100
                balance_info[sentiment] = f"{count} ({percentage:.1f}%)"
            
            print("dataset balance:")
            for sentiment, info in balance_info.items():
                print(f"  {sentiment}: {info}")

            # plot data distribution
            plt.figure(figsize=(8, 6))
            sns.countplot(x='sentiment', data=df)
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.savefig('images/sentiment_distribution.png')
            plt.close()
            
            return df
            
        except Exception as e:
            print(f"failed to read external dataset: {e}")
            raise ValueError(f"failed to load training dataset: {e}")

    # preprocess text
    def preprocess_text(self, text):
        # convert to lowercase
        text = text.lower()

        # remove special characters but keep some important punctuation marks
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)

        # tokenize
        tokens = word_tokenize(text)

        # remove stop words and lemmatize, but keep some important words
        important_words = {'not', 'no', 'never', 'very', 'too', 'more', 'most', 'best', 'worst'}
        processed_tokens = []
        for token in tokens:
            if token in important_words or (token not in self.stop_words and len(token) > 2):
                processed_tokens.append(self.lemmatizer.lemmatize(token))

        return ' '.join(processed_tokens)

    # prepare data for training
    def prepare_data(self, df):
        print("text preprocessing...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # check the number of samples for each class
        class_counts = df['sentiment_numeric'].value_counts().sort_index()
        print("number of samples for each class:")
        sentiment_names = ['negative', 'neutral', 'positive']
        for i, count in enumerate(class_counts):
            print(f"{sentiment_names[i]}: {count}")

        # if the number of samples is large enough, use stratified sampling; otherwise, do not use it
        use_stratify = all(count >= 2 for count in class_counts) and len(df) >= 10
        
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'],
                df['sentiment_numeric'],
                test_size=0.25,
                random_state=42,
                stratify=df['sentiment_numeric']
            )
        else:
            print("number of samples is not large enough, do not use stratified sampling")
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'],
                df['sentiment_numeric'],
                test_size=0.25,
                random_state=42
            )

        print(f"training set size: {len(X_train)}")
        print(f"test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, model_type='logreg'):
        # train sentiment analysis model
        if model_type == 'logreg':
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95)),
                ('classifier', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'))
            ])
        elif model_type == 'svm':
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95)),
                ('classifier', LinearSVC(C=1.0, class_weight='balanced', max_iter=2000))
            ])
        elif model_type == 'rf':
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=2, max_df=0.95)),
                ('classifier', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42))
            ])
        elif model_type == 'nb':
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95)),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
        else:
            raise ValueError("model type must be 'logreg', 'svm', 'rf' or 'nb'")

        # train model
        model.fit(X_train, y_train)
        self.model = model

        return model

    # Use grid search to find the best parameters
    def train_best_model_with_grid_search(self, X_train, y_train):
        print("Grid search optimization in progress...")
        
        # Defining the parameter grid
        param_grid = {
            'tfidf__max_features': [8000, 10000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'classifier__C': [0.5, 1.0, 2.0],
        }
        
        # Creating a base model
        base_model = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=2, max_df=0.95)),
            ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced'))
        ])
        
        # Grid Search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='f1_weighted',  # Use F1 score, suitable for imbalanced data
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Optimal parameters: {grid_search.best_params_}")
        print(f"Best cross validation score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        """
        evaluate the model performance
        """
        y_pred = model.predict(X_test)

        # calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # generate the classification report
        report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])

        # generate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)

        return accuracy, report, cm

    def plot_confusion_matrix(self, cm):
        """
        plot the confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('images/confusion_matrix.png')
        #plt.show()

    def save_model(self, filename='sentiment_model.pkl'):
        """
        save the trained model
        """
        if self.model is None:
            raise ValueError("model is not trained")

        joblib.dump(self.model, filename)
        print(f"model is saved as {filename}")

    def load_model(self, filename='sentiment_model.pkl'):
        """
        load the saved model
        """
        self.model = joblib.load(filename)
        print(f"model is loaded from {filename}")

    def predict(self, news_text):
        """
        predict the sentiment of the input news text
        """
        if self.model is None:
            raise ValueError("model is not trained or loaded")

        # preprocess text
        processed_text = self.preprocess_text(news_text)

        # predict sentiment
        sentiment = self.model.predict([processed_text])[0]

        # map to text label
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment_label = sentiment_map[sentiment]

        # get probability score (if the model supports)
        try:
            probabilities = self.model.predict_proba([processed_text])[0]
            confidence = probabilities.max()
        except:
            confidence = None

        return {
            'sentiment': sentiment,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
        }
    
    def prepare_model(self):
        # load external dataset
        df = self.load_external_dataset()
        
        print("\nprepare data...")
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        print("\n train multiple models for comparison...")
        models = {
            'Logistic Regression': 'logreg',
            'Support Vector Machine': 'svm',
            'Random Forest': 'rf',
            'Naive Bayes': 'nb'
        }

        results = {}

        for name, model_type in models.items():
            print(f"\ntrain {name} model...")
            model = self.train_model(X_train, y_train, model_type)

            # evaluate the model
            print(f"evaluate {name} model...")
            accuracy, report, cm = self.evaluate_model(model, X_test, y_test)
            results[name] = accuracy

            print(f"{name} accuracy: {accuracy:.4f}")
            print(f"{name} classification report:\n{report}")

        # select the best model
        best_model = max(results, key=results.get)
        print(f"\nthe best base model is {best_model}, accuracy is {results[best_model]:.4f}")

        # always use the optimized Logistic Regression as the final model (provide confidence score)
        print("\nuse grid search to optimize Logistic Regression as the final model...")
        optimized_model = self.train_best_model_with_grid_search(X_train, y_train)
        
        # evaluate the optimized model
        accuracy, report, cm = self.evaluate_model(optimized_model, X_test, y_test)
        print(f"\nthe final optimized model accuracy: {accuracy:.4f}")
        print(f"the final optimized model classification report:\n{report}")

        # save the final model
        self.save_model()

if __name__ == "__main__":
    analyzer = NewsSentimentAnalyzer()
    
    # check if the model file exists
    if not os.path.exists('sentiment_model_v2.pkl'):
        print("model file does not exist, start training model")
        analyzer.prepare_model()
    else:
        print("model file exists, load model directly")
        analyzer.load_model()


    # test the model
    demo_news = ''' 
"Sensing myself called to continue in this same path, I chose to take the name Leo XIV. There are different reasons for this, but mainly because Pope Leo XIII in his historic Encyclical 'Rerum Novarum' addressed the social question in the context of the first great industrial revolution," Leo XIV said. "In our own day, the Church offers to everyone the treasury of her social teaching in response to another industrial revolution and to developments in the field of artificial intelligence that pose new challenges for the defence of human dignity, justice, and labor."

Saturday's address isn't the first time the Catholic Church has reflected on artificial intelligence.

In January, the Holy See, the governing body of the Catholic Church, published a lengthy note on the relationship between artificial intelligence and human intelligence. The note said the Catholic Church "encourages the advancement of science, technology, the arts, and other forms of human endeavor" but sought to address the "anthropological and ethical challenges raised by AI — issues that are particularly significant, as one of the goals of this technology is to imitate the human intelligence that designed it."
'''

    result = analyzer.predict(demo_news)
    print("\ntest result:")
    print(result)

