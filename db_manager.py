import sqlite3
import json
import pandas as pd

class DBManager:
    def __init__(self, db_path='news_data.db'):
        """
        Initialize database manager
        
        Args:
        db_path (str): Database file path
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database, create necessary tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                date TEXT,
                sourcecountry TEXT,
                keywords TEXT,
                categories TEXT,
                sentiment TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_news(self, news_data):
        """
        Save news data to database
        
        Args:
        news_data (dict): Dictionary containing news analysis results
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT OR REPLACE INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    None,
                    news_data["url"],
                    news_data["title"],
                    news_data["date"],
                    news_data["sourcecountry"],
                    json.dumps(news_data["keywords"]),
                    json.dumps(news_data["categories"]),
                    news_data["sentiment"]
                )
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Failed to save news data: {str(e)}")
            return False
        finally:
            conn.close()
    
    def check_news_exists(self, url):
        """Check if news exists"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM news WHERE url = ?", (url,))
        count = c.fetchone()[0]
        return count > 0
    
    def get_all_news(self):
        """
        Get all news from database
        
        Returns:
        DataFrame: DataFrame containing all news data
        """
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM news", conn)
            if not df.empty:
                # Convert JSON strings to Python lists
                df["keywords"] = df["keywords"].apply(json.loads)
                df["categories"] = df["categories"].apply(json.loads)
            return df
        except Exception as e:
            print(f"Failed to get news data: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_news_by_country(self, country):
        """
        Get news by country
        
        Args:
        country (str): Country name
        
        Returns:
        DataFrame: DataFrame containing news from the specified country
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM news WHERE sourcecountry = ?"
            df = pd.read_sql_query(query, conn, params=(country,))
            if not df.empty:
                df["keywords"] = df["keywords"].apply(json.loads)
                df["categories"] = df["categories"].apply(json.loads)
            return df
        except Exception as e:
            print(f"Failed to get news data: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_news_by_sentiment(self, sentiment):
        """
        Get news by sentiment
        
        Args:
        sentiment (str): Sentiment type (positive, negative, neutral)
        
        Returns:
        DataFrame: DataFrame containing news with the specified sentiment
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM news WHERE sentiment = ?"
            df = pd.read_sql_query(query, conn, params=(sentiment,))
            if not df.empty:
                df["keywords"] = df["keywords"].apply(json.loads)
                df["categories"] = df["categories"].apply(json.loads)
            return df
        except Exception as e:
            print(f"Failed to get news data: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def clear_all_news(self):
        """
        Clear all news data from database
        
        Returns:
        bool: True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("DELETE FROM news")
            conn.commit()
            affected_rows = c.rowcount
            print(f"Successfully cleared {affected_rows} news data")
            return True
        except Exception as e:
            print(f"Failed to clear news data: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_news_count(self):
        """
        Get total number of news in database
        
        Returns:
        int: Number of news
        """
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM news")
            count = c.fetchone()[0]
            return count
        except Exception as e:
            print(f"Failed to get news count: {str(e)}")
            return 0
        finally:
            conn.close()

# Example usage
if __name__ == "__main__":
    db = DBManager()
    # Example data
    sample_news = {
        "url": "https://example.com/news/1",
        "title": "Example news title",
        "date": "20250505T084500Z",
        "sourcecountry": "China",
        "keywords": ["AI", "technology", "innovation"],
        "categories": ["technology", "innovation"],
        "sentiment": "positive"
    }
    db.save_news(sample_news)
    
    # Get all news
    all_news = db.get_all_news()
    print(f"Number of news in database: {len(all_news)}") 
