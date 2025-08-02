News Bias and Diversity Analysis System

This is a news analysis app built using Streamlit that analyzes the bias and diversity of news in different regions.

## Features

- Batch News Analysis: Search and analyze multiple news articles based on keywords
- Single News Analysis: Analyze the content of a single news URL
- Data Visualization: Display news sentiment, country of origin distribution, and category analysis
- Flexible expansion of multiple analyzers: Now supports machine learning analysis methods

## Running with Docker

### Prerequisites

- Install [Docker](https://docs.docker.com/get-docker/)
- Install [Docker Compose](https://docs.docker.com/compose/install/)

### Running with Docker Compose

1. Clone the repository
```bash
git clone <repository address>
cd <repository directory>
```

2. Create and configure the `.env` file
```bash
cp .env.example .env
```

3. Build and start the application using Docker Compose
```bash
docker-compose up -d
```

4. Access the application in a browser
```
http://localhost:8501
```

### Build directly using the Dockerfile

1. Build the Docker image
```bash
docker build -t news-analyzer .
```

2. Run the Docker container
```bash
docker run -d -p 8501:8501 --env-file .env --name news-analyzer news-analyzer
```

3. Access the application in a browser
```
http://localhost:8501
```

## Local development

1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app.py
```

## Environment Variables

The application requires the following environment variables:

- `LLM_API_KEY`: Language Model API key
- `LLM_MODEL_NAME`: Language Model name
- `LLM_BASE_URL`: Language Model API base URL

## Data Visualization

The application provides the following data visualizations:

- **News Data Table**: Displays news headlines, country of origin, and sentiment
- **Sentiment Analysis**: Displays the sentiment distribution of news (positive, negative, neutral)
- **Country Distribution**: Displays the distribution of news by country of origin
- **Category Analysis**: Displays the distribution of news categories
- **Time Trend**: Displays the time trend of news releases

## Project Structure

- `app.py`: Main application file, containing the Streamlit interface
- `news_analyzer.py`: News acquisition and analysis module
- `db_manager.py`: Database management module
- `visualizer.py`: Data visualization module
- `analyzers/`: Contains various analyzer implementations

- `model/classifierAnalyzer.py`: Classification model
- `model/keywordsAnalyzer.py`: Keyword extraction
- `analyzers/model/sentimentAnalyzer.py`: Sentiment analysis model
- `python_analyzer.py`: Machine learning model management class
- `analyzer_factory.py`: Analyzer factory class
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (need to be created by yourself)

## Analyzer Description

### Machine Learning Model Analyzer
This analysis method is implemented in pure Python, based on NLTK and scikit-learn. This method does not require an API key and is completely free to use, but accuracy may be lower. Local model training is supported to improve accuracy.

## Technology Stack

- Python
- Streamlit
- OpenAI API
- HuggingFace API
- NLTK
- scikit-learn
- SQLite
- Pandas
- Matplotlib
- BeautifulSoup
