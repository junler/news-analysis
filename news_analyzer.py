import requests
import json
import os
from dotenv import load_dotenv
from analyzers.analyzer_factory import AnalyzerFactory
from tools import get_text_from_url
from urllib.parse import urlencode

# Load environment variables
load_dotenv()

class NewsAnalyzer:
    def __init__(self, analyzer_type=None):
        """
        Initialize news analyzer
        
        Args:
        analyzer_type (str): Analyzer type, optional values are 'openai', 'huggingface', 'machine_learning'
        """
        self.api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.analyzer_type = analyzer_type
        # Use factory to create analyzer
        self.analyzer = AnalyzerFactory.get_analyzer(analyzer_type)
    
    def fetch_news(self, keyword=None, country=None, timespan="7d", max_records=50):
        """
        Fetch news data through GDELT API
        
        Args:
        keyword (str): Search keyword
        timespan (str): Time range (1d, 3d, 7d, 14d, 30d)
        max_records (int): Maximum number of records
        
        Returns:
        dict: JSON response containing news articles
        """
        query = f'sourcelang:english'
        if country:
            query += f' AND sourcecountry:{country}'
        if keyword:
            query += f' AND {keyword}'

        params = {
            "query": query,
            "mode": "artlist",
            "timespan": timespan,
            "format": "json",
            "maxrecords": max_records
        }
        # Print request url
        print(f"Request url: {self.api_url}?{urlencode(params)}")
        headers = {
            "User-Agent": "python-requests/2.31.0",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
            "Connection": "close"  # Disable Keep-Alive, reduce connection reuse
        }
        try:
            response = requests.get(self.api_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Check if response content is empty
                if not response.content:
                    print("API returned empty content")
                    return None
                
                # Check if Content-Type is JSON
                content_type = response.headers.get('content-type', '').lower()
                if 'application/json' not in content_type:
                    print(f"API returned non-JSON format content, Content-Type: {content_type}")
                    print(f"Response content first 100 characters: {response.text[:100]}")
                    return None
                
                # Try to parse JSON
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    print(f"Response status code: {response.status_code}")
                    print(f"Response content first 200 characters: {response.text[:200]}")
                    return None
            else:
                print(f"Failed to get news, status code: {response.status_code}")
                print(f"Error information: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timeout, please check network connection")
            return None
        except requests.exceptions.ConnectionError:
            print("Connection error, please check network connection")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return None
    
    def analyze_content(self, url, title, date, sourcecountry):
        # If the analyzer is machine_learning, it needs to call ai to get the text in the url
        print("analyze_content-analyzer_type: ", self.analyzer_type)
        if self.analyzer_type == "machine_learning":
            print(f"Use machine_learning analyzer, first get the text in the url {url}")
            url = get_text_from_url(url)
            # print("Get the text in the url: ", url)
            # If the returned value starts with Sorry, I'm unable, ignore case, return None
            skip_words = ["sorry", "i'm unable"]
            if not url or any(word.lower() in url.lower() for word in skip_words):
                print("Skip this news, because the url contains skip_words")
                return None
        """
        Analyze news content
        
        Args:
        url (str): News URL
        title (str): News title
        date (str): News date
        sourcecountry (str): Source country
        
        Returns:
        dict: Dictionary containing analysis results
        """
        # Use the selected analyzer to analyze
            
        return self.analyzer.analyze(url, title, date, sourcecountry)

    def set_analyzer(self, analyzer_type):
        """
        Set analyzer type
        
        Args:
        analyzer_type (str): Analyzer type, optional values are 'openai', 'huggingface', 'machine_learning'
        """
        self.analyzer_type = analyzer_type
        self.analyzer = AnalyzerFactory.get_analyzer(analyzer_type)
        return self.analyzer
    
    def get_model_info(self):
        """
        Get current analyzer model information
        
        Returns:
        dict: Dictionary containing analyzer type and model name
        """
        info = {
            "analyzer_type": self.analyzer_type,
            "sentiment_model": "Unknown",
            "classification_model": "Unknown"
        }
        
        try:
            if self.analyzer_type == "openai":
                info["sentiment_model"] = getattr(self.analyzer, 'model_name', 'Unknown')
                info["classification_model"] = getattr(self.analyzer, 'model_name', 'Unknown')
            elif self.analyzer_type == "huggingface":
                info["sentiment_model"] = getattr(self.analyzer, 'sentiment_model', 'Unknown')
                info["classification_model"] = getattr(self.analyzer, 'classification_model', 'Unknown')
            elif self.analyzer_type == "machine_learning":
                info["sentiment_model"] = "Sentiment Model (Logistic Regression)"
                info["classification_model"] = "Classification Model (Random Forest)"
        except Exception as e:
            print(f"Failed to get model information: {e}")
        
        return info

# Example usage
if __name__ == "__main__":
    analyzer = NewsAnalyzer()
    news = analyzer.fetch_news("artificial intelligence", "CH", "1d", 5)
    
    if news and "articles" in news:
        for article in news["articles"]:
            print(f"Title: {article['title']}")
            print(f"URL: {article['url']}")
            print(f"Source country: {article.get('sourcecountry', 'Unknown')}")
            print("---") 
            analyzer.analyze_content(article['url'], article['title'], article.get('seendate', ''), article.get('sourcecountry', ''))
            break
