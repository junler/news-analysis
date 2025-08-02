import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AnalyzerFactory:
    """Analyzer factory class, used to create different types of news analyzers"""
    
    @staticmethod
    def get_analyzer(analyzer_type=None):
        """
        Get analyzer of specified type
        
        Args:
        analyzer_type (str): Analyzer type, optional values are 'openai', 'huggingface', 'machine_learning',
                            if None, use environment variable or default value
        
        Returns:
        object: Analyzer instance
        """
        # If no type is specified, get from environment variable, default is 'openai'
        if analyzer_type is None:
            analyzer_type = os.getenv("DEFAULT_ANALYZER", "machine_learning")
        
        analyzer_type = analyzer_type.lower()
        
        if analyzer_type == "openai":
            from analyzers.openai_analyzer import OpenAIAnalyzer
            return OpenAIAnalyzer()
        elif analyzer_type == "huggingface":
            from analyzers.huggingface_analyzer import HuggingFaceAnalyzer
            return HuggingFaceAnalyzer()
        elif analyzer_type == "machine_learning":
            from analyzers.python_analyzer import PythonAnalyzer
            return PythonAnalyzer()
        else:
            raise ValueError(f"Invalid analyzer type: {analyzer_type}")
