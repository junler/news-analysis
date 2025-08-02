from  analyzers.python_analyzer import PythonAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


analyzer = PythonAnalyzer()
demo_news = ''' 
"Sensing myself called to continue in this same path, I chose to take the name Leo XIV. There are different reasons for this, but mainly because Pope Leo XIII in his historic Encyclical 'Rerum Novarum' addressed the social question in the context of the first great industrial revolution," Leo XIV said. "In our own day, the Church offers to everyone the treasury of her social teaching in response to another industrial revolution and to developments in the field of artificial intelligence that pose new challenges for the defence of human dignity, justice, and labor."

Saturday's address isn't the first time the Catholic Church has reflected on artificial intelligence.

In January, the Holy See, the governing body of the Catholic Church, published a lengthy note on the relationship between artificial intelligence and human intelligence. The note said the Catholic Church "encourages the advancement of science, technology, the arts, and other forms of human endeavor" but sought to address the "anthropological and ethical challenges raised by AI — issues that are particularly significant, as one of the goals of this technology is to imitate the human intelligence that designed it."
'''
result = analyzer.analyze(demo_news)
print(result)

def test_keyword_frequency(result):
    # Generate keyword frequency table
    keyword_freq = Counter(result['keywords'])
    top_keywords = keyword_freq.most_common(10)

    # (a) Word cloud visualization
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Keyword WordCloud (YAKE)")
    plt.show()

    # (b) Keyword frequency bar chart
    keywords, freqs = zip(*top_keywords)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(freqs), y=list(keywords), palette='viridis')
    plt.xlabel("Frequency")
    plt.ylabel("Keyword")
    plt.title("Top 10 Keywords by Frequency (YAKE)")
    plt.show()
