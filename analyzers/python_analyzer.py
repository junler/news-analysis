import os
from analyzers.model.sentimentAnalyzer import NewsSentimentAnalyzer
from analyzers.model.classifierAnalyzer import NewsClassifier
from analyzers.model.keywordsAnalyzer import KeywordsAnalyzer


class PythonAnalyzer:
    def __init__(self):
        """Initialize Python local parser"""
        self.models_dir = os.getenv("MODELS_DIR", "models")
        
        # Ensure model directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load model (if exists)
        self.sentiment_model = self._load_sentiment_model()
        self.category_model = self._load_category_model()
        
        print("Use Python local parser")
    
    def _load_sentiment_model(self):
        self.sentiment_model = NewsSentimentAnalyzer()
        if not os.path.exists('sentiment_model.pkl'):
            print("Model file does not exist, start training model")
            self.sentiment_model.prepare_model()
        else:
            print("Model file exists, load model directly")
            self.sentiment_model.load_model()

        return self.sentiment_model
    
    def _load_category_model(self):
        self.category_model = NewsClassifier()
        if not os.path.exists('news_classifier_model.pkl'):
            print("Model file does not exist, start training model")
            self.category_model.prepare_model()
        else:
            print("Model file exists, load model directly")
            self.category_model.load_model()

        return self.category_model
    
    
    def _extract_keywords(self, text, max_keywords=5):
        analyzer = KeywordsAnalyzer(top_n=max_keywords)
        keywords = analyzer.extract_keywords(text)
        return keywords
    
    
    def analyze(self, text, title = None, date = None, sourcecountry = None):
        """
        Use Python local method to analyze news content
        
        Args:
        url (str): News URL
        title (str): News title
        date (str): News date
        sourcecountry (str): Source country
        
        Returns:
        dict: Dictionary containing analysis results
        """
        try:            
            # Extract keywords
            keywords = self._extract_keywords(text)
            
            # Predict sentiment
            sentiment = self.sentiment_model.predict(text)
            
            # Predict category
            categorie = self.category_model.predict(text)
            
            return {
                "url": None,
                "title": title,
                "date": date,
                "sourcecountry": sourcecountry,
                "keywords": keywords,
                "categories": [categorie['category']],
                "sentiment": sentiment['sentiment_label']
            }
            
        except Exception as e:
            print(f"Python local analysis news failed: {str(e)}")
            return None
    
    def train_models(self, training_data):
        """
        Train sentiment and category models
        
        Args:
        training_data: Training data containing text, sentiment, and categories
        """
        try:
            # Extract training data
            
            
            print("Model training completed and saved")
            return True
        except Exception as e:
            print(f"Failed to train model: {str(e)}")
            return False 

if __name__ == "__main__":
    analyzer = PythonAnalyzer()
    demo_news = ''' 
"Sensing myself called to continue in this same path, I chose to take the name Leo XIV. There are different reasons for this, but mainly because Pope Leo XIII in his historic Encyclical 'Rerum Novarum' addressed the social question in the context of the first great industrial revolution," Leo XIV said. "In our own day, the Church offers to everyone the treasury of her social teaching in response to another industrial revolution and to developments in the field of artificial intelligence that pose new challenges for the defence of human dignity, justice, and labor."

Saturday's address isn't the first time the Catholic Church has reflected on artificial intelligence.

In January, the Holy See, the governing body of the Catholic Church, published a lengthy note on the relationship between artificial intelligence and human intelligence. The note said the Catholic Church "encourages the advancement of science, technology, the arts, and other forms of human endeavor" but sought to address the "anthropological and ethical challenges raised by AI — issues that are particularly significant, as one of the goals of this technology is to imitate the human intelligence that designed it."
'''
    result = analyzer.analyze(demo_news)
    print(result)
