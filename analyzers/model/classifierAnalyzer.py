import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os

# download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class NewsClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
        # initialize lemmatizer, with exception handling
        try:
            self.lemmatizer = WordNetLemmatizer()
            # test if WordNet is working
            test_word = self.lemmatizer.lemmatize("running")
            self.use_lemmatizer = True
            print("using WordNet lemmatizer")
        except Exception as e:
            print(f"WordNet initialization failed: {e}")
            print("using Porter stemmer as backup")
            self.lemmatizer = PorterStemmer()
            self.use_lemmatizer = False
            
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("stop words download failed, using default stop words")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            
        self.categories = None
        # dataset category mapping
        self.class_mapping = {
            1: 'World',
            2: 'Sports', 
            3: 'Business',
            4: 'Technology'
        }

    # load AG-News dataset
    def load_data(self, train_path=None, test_path=None, extended_path=None):
        if train_path is None:
            train_path = "data/classifier/train.csv"
        if test_path is None:
            test_path = "data/classifier/test.csv"
            
        try:
            # load training set
            train_df = pd.read_csv(train_path, names=['class_index', 'title', 'description'], header=0)
            print(f"AG-News training set successfully loaded! number of samples: {len(train_df)}")
            
            # load test set
            test_df = pd.read_csv(test_path, names=['class_index', 'title', 'description'], header=0)
            print(f"AG-News test set successfully loaded! number of samples: {len(test_df)}")
            
            # merge title and description, create full text
            train_df['text'] = train_df['title'] + ' ' + train_df['description']
            test_df['text'] = test_df['title'] + ' ' + test_df['description']
            
            # map category labels
            train_df['category'] = train_df['class_index'].map(self.class_mapping)
            test_df['category'] = test_df['class_index'].map(self.class_mapping)
            
            # merge training set and test set
            ag_news_df = pd.concat([train_df, test_df], ignore_index=True)
            
            print(f"AG-News total dataset number of samples: {len(ag_news_df)}")
            print("AG-News category distribution:")
            #print(ag_news_df['category'].value_counts())
            print(ag_news_df.head(6))

            print("\ndata info: \n")
            print(ag_news_df.info())
            print("\ndata null: \n")
            print(ag_news_df.isnull().sum())
            print("\ndata describe: \n")
            print(ag_news_df.describe())

            return ag_news_df[['text', 'category']]
            
            # load extended categories dataset for future
            try:
                extended_df = pd.read_csv(extended_path)
                print(f"\nextended dataset successfully loaded! number of samples: {len(extended_df)}")
                print("extended categories distribution:")
                print(extended_df['category'].value_counts())
                
                # merge AG-News data and extended categories data
                combined_df = pd.concat([ag_news_df[['text', 'category']], extended_df], ignore_index=True)
                
                # shuffle dataset
                combined_df = combined_df.sample(frac=1).reset_index(drop=True)
                print(combined_df.head(10))
                print(f"\ntotal dataset number of samples after merging: {len(combined_df)}")
                print("merged categories distribution:")
                print(combined_df['category'].value_counts())
                
                return combined_df
                
            except Exception as e:
                print(f"failed to load extended dataset: {e}")
                print("using AG-News dataset only...")
                return ag_news_df[['text', 'category']]
                
        except Exception as e:
            print(f"failed to load AG-News dataset: {e}")
            # if cannot load AG-News file, only use extended categories dataset
            try:
                extended_df = pd.read_csv(extended_path)
                print(f"using extended dataset... number of samples: {len(extended_df)}")
                return extended_df
            except Exception as e2:
                print(f"failed to load extended dataset: {e2}")
                raise Exception("cannot load any dataset file!")

    def preprocess_text(self, text):
        """text preprocessing function"""
        try:
            # convert to lowercase
            text = text.lower()

            # remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # tokenization
            try:
                words = nltk.word_tokenize(text)
            except:
                # if NLTK tokenization fails, use simple split
                words = text.split()

            # remove stop words and lemmatize
            processed_words = []
            for word in words:
                if word not in self.stop_words and len(word) > 2:
                    try:
                        if self.use_lemmatizer:
                            processed_word = self.lemmatizer.lemmatize(word)
                        else:
                            processed_word = self.lemmatizer.stem(word)
                        processed_words.append(processed_word)
                    except Exception as e:
                        # if lemmatization fails, use original word
                        processed_words.append(word)

            return ' '.join(processed_words)
            
        except Exception as e:
            print(f"text preprocessing error: {e}")
            # return basic cleaned text
            return re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # explore data
    def explore_data(self, df):
        print(f"data shape: {df.shape}")
        print("\ndata category distribution:")
        category_counts = df['category'].value_counts()
        print(category_counts)

        # plot data distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='category', data=df)
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.savefig('images/category_distribution.png')
        plt.close()

        # analyze text length
        df['text_length'] = df['text'].apply(len)
        print("\ntext length statistics:")
        print(df['text_length'].describe())

        # analyze average text length for each category
        print("\naverage text length for each category:")
        avg_length_by_category = df.groupby('category')['text_length'].mean().sort_values(ascending=False)
        print(avg_length_by_category)

        # plot average text length
        plt.figure(figsize=(8, 6))
        sns.barplot(x=avg_length_by_category.index, y=avg_length_by_category.values)
        plt.title('Average Text Length for Each Category')
        plt.xlabel('Category')
        plt.ylabel('Average Text Length')
        plt.savefig('images/average_text_length_by_category.png')

        return df

    # prepare training and testing data
    def prepare_data(self, df):
        print("\npreprocessing text...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # check the number of samples for each category
        category_counts = df['category'].value_counts()
        print("\n number of samples for each category:")
        print(category_counts)

        # filter out categories with less than 2 samples
        min_samples = 2
        valid_categories = category_counts[category_counts >= min_samples].index
        if len(valid_categories) < len(category_counts):
            print(f"\nwarning: removed categories with less than {min_samples} samples")
            print(f"removed categories: {set(category_counts.index) - set(valid_categories)}")
            df = df[df['category'].isin(valid_categories)]
            print(f"filtered dataset size: {len(df)}")

        # split training set and test set
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'],
                df['category'],
                test_size=0.2,
                random_state=42,
                stratify=df['category']
            )
        except ValueError as e:
            print(f"\nwarning: cannot perform stratified sampling, using normal random sampling: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'],
                df['category'],
                test_size=0.2,
                random_state=42
            )

        print(f"training set size: {len(X_train)}")
        print(f"test set size: {len(X_test)}")

        # feature extraction
        print("extracting features...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        return X_train_tfidf, X_test_tfidf, y_train, y_test


    # train multiple models and return the best model
    def train_models(self, X_train, y_train):
        print("\ntraining models...")

        models = {
            #"naive bayes": MultinomialNB(),
            #"logistic regression": LogisticRegression(max_iter=1000),
            #"support vector machine": LinearSVC(max_iter=10000),
            "random forest": RandomForestClassifier(n_estimators=100)
        }

        best_accuracy = 0
        best_model_name = None

        for name, model in models.items():
            print(f"\ntraining {name} model...")
            model.fit(X_train, y_train)
            self.model = model

            # evaluate on training set
            train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)
            print(f"{name} - training set accuracy: {train_accuracy:.4f}")

            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy
                best_model_name = name
                self.model = model

        print(f"\nselected {best_model_name} as the best model, training set accuracy: {best_accuracy:.4f}")
        return self.model

    # evaluate model performance
    def evaluate_model(self, X_test, y_test):
        print("\nmodel evaluation:")

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"test set accuracy: {accuracy:.4f}")

        print("\n category report:")
        report = classification_report(y_test, y_pred)
        print(report)

        # plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        self.categories = sorted(list(set(y_test)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.categories,
                    yticklabels=self.categories)
        plt.title('confusion matrix')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.tight_layout()
        plt.savefig('images/confusion_matrix_classifier.png')
        print("confusion matrix is saved as 'confusion_matrix_classifier.png'")

    def predict(self, news_text):
        """predict news text category"""
        if self.model is None or self.vectorizer is None:
            print("error: model not trained!")
            return None

        try:
            # preprocess text
            processed_text = self.preprocess_text(news_text)

            # feature extraction - use the same vectorizer
            text_tfidf = self.vectorizer.transform([processed_text])
                
            # debug information
            print(f"number of features after vectorization: {text_tfidf.shape[1]}")
            print(f"expected number of features: {self.vectorizer.max_features}")

            # predict
            prediction = self.model.predict(text_tfidf)[0]

            # get prediction probabilities for each category (if model supports)
            probabilities = {}
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(text_tfidf)[0]
                for i, category in enumerate(self.model.classes_):
                    probabilities[category] = proba[i]

            return {
                'category': prediction,
                'probabilities': probabilities
            }
            
        except Exception as e:
            print(f"error during prediction: {e}")
            print(f"error type: {type(e).__name__}")
            
            # provide more debug information
            if hasattr(self, 'vectorizer') and self.vectorizer is not None:
                print(f"vectorizer vocabulary size: {len(self.vectorizer.vocabulary_)}")
                print(f"vectorizer max features: {self.vectorizer.max_features}")
            
            return None

    def save_model(self, vectorizer_path='vectorizer.pkl', model_path='news_classifier_model.pkl'):
        """save trained model and vectorizer"""
        import pickle

        # save vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        # save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"\nmodel and vectorizer saved to '{model_path}' and '{vectorizer_path}'")

    def load_model(self, vectorizer_path='vectorizer.pkl', model_path='news_classifier_model.pkl'):
        """load saved model and vectorizer"""
        import pickle

        try:
            # load vectorizer
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

            # load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            print(f"\nsuccessfully loaded model and vectorizer!")
            return True
        except Exception as e:
            print(f"failed to load model: {e}")
            return False
    
    def prepare_model(self):
        df = self.load_data()

        # explore data
        self.explore_data(df)

        # prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # train model
        self.train_models(X_train, y_train)

        # evaluate model
        self.evaluate_model(X_test, y_test)

        # save model
        self.save_model()

if __name__ == "__main__":
    # create classifier instance
    classifier = NewsClassifier()
    
    if not os.path.exists('news_classifier_model_v2.pkl'):
        # load data
        print("loading AG-News dataset and extended categories dataset, and training model...")
        classifier.prepare_model()
    else:
        print("model file exists, directly load model")
        classifier.load_model()

    # test different categories of news classification
    test_news_samples = [
        {
            'text': '''
    WASHINGTON (AP) — When Meghan Sells heads to Providence Park to watch Oregon's professional women's soccer team, she finds herself among a fairly mixed crowd — groups of young women, dads bringing their children, youth players checking out the Thorns' latest match.
            The physician's assistant is a self-described lifelong sports fan and former softball player who "will watch any sport." That includes both collegiate and professional sports for women, putting Sells squarely in a fan base that suddenly has more options than ever before and is seen as fertile ground for teams and advertisers eager to ride the rising interest in the women's game.
            ''',
            'expected': 'Sports'
        },
        {
            'text': '''
            Apple Inc. announced today the launch of its new iPhone model featuring advanced artificial intelligence capabilities and improved battery life. The tech giant expects strong sales in the upcoming quarter as consumers upgrade to the latest technology.
            ''',
            'expected': 'Technology'
        },
        {
            'text': '''
            Doctors at Johns Hopkins Hospital have discovered a new treatment for diabetes that shows promising results in clinical trials. The research could potentially help millions of patients worldwide manage their condition more effectively.
            ''',
            'expected': 'Health'
        },
        {
            'text': '''
            The Supreme Court will hear arguments next month on a landmark case involving voting rights that could significantly impact future elections across the United States.
            ''',
            'expected': 'Politics'
        }
    ]
    
    print("\n=== news classification test results ===")
    for i, sample in enumerate(test_news_samples, 1):
        prediction = classifier.predict(sample['text'])
        print(f"\ntest sample {i} (expected category: {sample['expected']}):")
        if prediction:
            print(f"prediction category: {prediction['category']}")
            print(f"prediction accuracy: {'✓' if prediction['category'] == sample['expected'] else '✗'}")
            if prediction['probabilities']:
                print("prediction probabilities:")
                for category, prob in sorted(prediction['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {category}: {prob:.4f}")
        else:
            print("prediction failed") 
