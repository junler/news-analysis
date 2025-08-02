import matplotlib.pyplot as plt
import pandas as pd
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: wordcloud module not installed, word cloud functionality will be unavailable")
import numpy as np
from collections import Counter
import matplotlib
import platform
import os

# Set matplotlib to support English text properly
def set_matplotlib_english():
    # Set global font to support English text
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    # Fix negative sign display issue
    plt.rcParams['axes.unicode_minus'] = False

# Apply English font settings
set_matplotlib_english()

class NewsVisualizer:
    def __init__(self):
        """Initialize news visualizer"""
        pass
    
    def generate_wordcloud(self, keywords_list):
        """
        Generate keywords word cloud
        
        Args:
        keywords_list (list): List of keyword lists
        
        Returns:
        fig: Matplotlib figure object
        """
        if not WORDCLOUD_AVAILABLE:
            # Create a notification chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Need to install wordcloud module\npip install wordcloud', 
                   ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.axis('off')
            return fig
        
        if not keywords_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No keywords data', ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        # Flatten keywords list and count
        all_keywords = [keyword for keywords in keywords_list for keyword in keywords]
        keyword_freq = Counter(all_keywords)
        
        if not keyword_freq:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No valid keywords', ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate_from_frequencies(keyword_freq)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Keywords Word Cloud', fontsize=16, pad=20)
        
        return fig
    
    def generate_wordcloud_by_country(self, df, country):
        """
        Generate keywords word cloud for specified country
        
        Args:
        df (DataFrame): DataFrame containing news data
        country (str): Country name
        
        Returns:
        fig: Matplotlib figure object
        """
        # Filter news for specified country
        country_df = df[df['sourcecountry'] == country]
        
        if country_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No news data from {country}', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        # Get keywords list
        keywords_list = country_df['keywords'].tolist()
        
        # Generate word cloud
        return self.generate_wordcloud(keywords_list)
    
    def generate_wordcloud_comparison(self, df, max_countries=None):
        """
        Generate multi-country keywords word cloud comparison
        
        Args:
        df (DataFrame): DataFrame containing news data
        max_countries (int): Maximum number of countries to display, show all if None
        
        Returns:
        fig: Matplotlib figure object
        """
        if not WORDCLOUD_AVAILABLE:
            # Create a notification chart
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.text(0.5, 0.5, 'Need to install wordcloud module\npip install wordcloud', 
                   ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.axis('off')
            return fig
        
        # Get news count for all countries
        country_counts = df['sourcecountry'].value_counts()
        if max_countries is not None:
            country_counts = country_counts.head(max_countries)
        countries = country_counts.index.tolist()
        
        if len(countries) < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Need at least 2 countries for comparison', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        # Calculate subplot layout - adjust for more countries
        n_countries = len(countries)
        if n_countries <= 4:
            cols = 2
        elif n_countries <= 9:
            cols = 3
        else:
            cols = 4
        rows = (n_countries + cols - 1) // cols
        
        # Dynamically adjust figure size
        fig_width = min(20, max(12, cols * 4))
        fig_height = min(24, max(8, rows * 4))
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, country in enumerate(countries):
            # Get news data for this country
            country_df = df[df['sourcecountry'] == country]
            keywords_list = country_df['keywords'].tolist()
            
            # Flatten keywords list and count
            all_keywords = [keyword for keywords in keywords_list for keyword in keywords if keywords]
            
            if all_keywords:
                keyword_freq = Counter(all_keywords)
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    max_words=50,
                    relative_scaling=0.5,
                    colormap='viridis'
                ).generate_from_frequencies(keyword_freq)
                
                # Display word cloud
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{country} ({len(country_df)} articles)', fontsize=12, pad=10)
                axes[i].axis('off')
            else:
                # No keywords case
                axes[i].text(0.5, 0.5, f'{country}\nNo keywords data', 
                           ha='center', va='center', fontsize=12)
                axes[i].set_title(f'{country} ({len(country_df)} articles)', fontsize=12, pad=10)
                axes[i].axis('off')
        
        # Hide extra subplots
        for j in range(n_countries, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Multi-Country News Keywords Word Cloud Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        return fig
    
    def generate_sentiment_chart(self, df):
        """
        Generate sentiment analysis bar chart
        
        Args:
        df (DataFrame): DataFrame containing news data
        
        Returns:
        fig: Matplotlib figure object
        """
        sentiment_counts = df['sentiment'].value_counts()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set English labels
        labels = []
        for idx in sentiment_counts.index:
            if idx == 'positive':
                labels.append('Positive')
            elif idx == 'negative':
                labels.append('Negative')
            elif idx == 'neutral':
                labels.append('Neutral')
            else:
                labels.append(idx)
        
        # Draw bar chart
        bars = ax.bar(labels, sentiment_counts.values, color=['green', 'red', 'blue'])
        
        # Add labels and title
        ax.set_title('News Sentiment Analysis', fontsize=15)
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{int(height)}',
                ha='center'
            )
        
        return fig
    
    def generate_sentiment_comparison_chart(self, df):
        """
        Generate multi-country sentiment analysis comparison chart
        
        Args:
        df (DataFrame): DataFrame containing news data
        
        Returns:
        fig: Matplotlib figure object
        """
        # Get sentiment distribution data for each country
        country_sentiment = df.groupby(['sourcecountry', 'sentiment']).size().unstack(fill_value=0)
        
        # Add missing sentiment types as 0
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment not in country_sentiment.columns:
                country_sentiment[sentiment] = 0
        
        # Reorder columns
        country_sentiment = country_sentiment[['positive', 'negative', 'neutral']]
        
        # Create figure - single chart layout
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Absolute count comparison
        x = np.arange(len(country_sentiment.index))
        width = 0.25
        
        bars1 = ax.bar(x - width, country_sentiment['positive'], width, 
                       label='Positive', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x, country_sentiment['negative'], width, 
                       label='Negative', color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + width, country_sentiment['neutral'], width, 
                       label='Neutral', color='#95a5a6', alpha=0.8)
        
        ax.set_title('Multi-Country News Sentiment Analysis - Absolute Count Comparison', fontsize=16, pad=20)
        ax.set_xlabel('Country', fontsize=12)
        ax.set_ylabel('News Count', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(country_sentiment.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def generate_sentiment_heatmap(self, df):
        """
        Generate sentiment analysis heatmap
        
        Args:
        df (DataFrame): DataFrame containing news data
        
        Returns:
        fig: Matplotlib figure object
        """
        # Get sentiment distribution percentage for each country
        country_sentiment = df.groupby(['sourcecountry', 'sentiment']).size().unstack(fill_value=0)
        
        # Add missing sentiment types as 0
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment not in country_sentiment.columns:
                country_sentiment[sentiment] = 0
        
        # Reorder columns and calculate percentage
        country_sentiment = country_sentiment[['positive', 'negative', 'neutral']]
        country_sentiment_pct = country_sentiment.div(country_sentiment.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Rename column labels to English
        country_sentiment_pct.columns = ['Positive', 'Negative', 'Neutral']
        
        # Create heatmap
        im = ax.imshow(country_sentiment_pct.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set tick labels
        ax.set_xticks(range(len(country_sentiment_pct.columns)))
        ax.set_xticklabels(country_sentiment_pct.columns)
        ax.set_yticks(range(len(country_sentiment_pct.index)))
        ax.set_yticklabels(country_sentiment_pct.index)
        
        # Add value labels
        for i in range(len(country_sentiment_pct.index)):
            for j in range(len(country_sentiment_pct.columns)):
                text = ax.text(j, i, f'{country_sentiment_pct.iloc[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
        
        ax.set_title('Multi-Country News Sentiment Analysis Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        return fig
    
    def generate_country_chart(self, df, top_n=10):
        """
        Generate country distribution bar chart
        
        Args:
        df (DataFrame): DataFrame containing news data
        top_n (int): Display top N countries
        
        Returns:
        fig: Matplotlib figure object
        """
        country_counts = df['sourcecountry'].value_counts().head(top_n)
        
        # Dynamically adjust chart size to fit number of countries
        width = max(14, min(24, len(country_counts) * 1.2))
        height = max(8, min(12, len(country_counts) * 0.4))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))
        bars = ax.bar(country_counts.index, country_counts.values, color='skyblue')
        
        # Add labels and title
        ax.set_title('News Source Country Distribution', fontsize=15)
        ax.set_xlabel('Country')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{int(height)}',
                ha='center'
            )
        
        plt.tight_layout()
        return fig
    
    def generate_category_chart(self, df):
        """
        Generate news category distribution chart
        
        Args:
        df (DataFrame): DataFrame containing news data
        
        Returns:
        fig: Matplotlib figure object
        """
        # Flatten category list and count
        all_categories = [cat for cats in df['categories'] for cat in cats]
        category_counter = Counter(all_categories)
        top_categories = dict(category_counter.most_common(10))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(top_categories.keys(), top_categories.values(), color='lightgreen')
        
        # Add labels and title
        ax.set_title('News Category Distribution', fontsize=15)
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{int(height)}',
                ha='center'
            )
        
        plt.tight_layout()
        return fig
    
    def generate_category_comparison_chart(self, df, top_n_categories=8):
        """
        Generate multi-country news category analysis comparison chart
        
        Args:
        df (DataFrame): DataFrame containing news data
        top_n_categories (int): Display top N most popular categories
        
        Returns:
        fig: Matplotlib figure object
        """
        # Expand category data - each news may have multiple categories
        expanded_data = []
        for _, row in df.iterrows():
            for category in row['categories']:
                expanded_data.append({
                    'sourcecountry': row['sourcecountry'],
                    'category': category
                })
        
        if not expanded_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No category data', ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        expanded_df = pd.DataFrame(expanded_data)
        
        # Get most popular categories
        top_categories = expanded_df['category'].value_counts().head(top_n_categories).index.tolist()
        
        # Filter data, keep only popular categories
        filtered_df = expanded_df[expanded_df['category'].isin(top_categories)]
        
        # Group by country and category
        country_category = filtered_df.groupby(['sourcecountry', 'category']).size().unstack(fill_value=0)
        
        # Ensure all popular categories exist
        for category in top_categories:
            if category not in country_category.columns:
                country_category[category] = 0
        
        # Reorder columns
        country_category = country_category[top_categories]
        
        # Create figure - single chart layout
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Absolute count comparison
        x = np.arange(len(country_category.index))
        width = 0.8 / len(top_categories)
        
        # Generate colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
        
        bars_list = []
        for i, category in enumerate(top_categories):
            bars = ax.bar(x + i * width - (len(top_categories) - 1) * width / 2, 
                          country_category[category], width, 
                          label=category, color=colors[i], alpha=0.8)
            bars_list.append(bars)
        
        ax.set_title('Multi-Country News Category Analysis - Absolute Count Comparison', fontsize=16, pad=20)
        ax.set_xlabel('Country', fontsize=12)
        ax.set_ylabel('News Count', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(country_category.index, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in bars_list:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', fontsize=8, rotation=90)
        
        plt.tight_layout()
        return fig
    
    def generate_category_heatmap(self, df, top_n_categories=10):
        """
        Generate category analysis heatmap
        
        Args:
        df (DataFrame): DataFrame containing news data
        top_n_categories (int): Display top N most popular categories
        
        Returns:
        fig: Matplotlib figure object
        """
        # Expand category data
        expanded_data = []
        for _, row in df.iterrows():
            for category in row['categories']:
                expanded_data.append({
                    'sourcecountry': row['sourcecountry'],
                    'category': category
                })
        
        if not expanded_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No category data', ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        expanded_df = pd.DataFrame(expanded_data)
        
        # Get most popular categories
        top_categories = expanded_df['category'].value_counts().head(top_n_categories).index.tolist()
        
        # Filter data
        filtered_df = expanded_df[expanded_df['category'].isin(top_categories)]
        
        # Group by country and category and calculate percentage
        country_category = filtered_df.groupby(['sourcecountry', 'category']).size().unstack(fill_value=0)
        
        # Ensure all popular categories exist
        for category in top_categories:
            if category not in country_category.columns:
                country_category[category] = 0
        
        # Reorder columns and calculate percentage
        country_category = country_category[top_categories]
        country_category_pct = country_category.div(country_category.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        im = ax.imshow(country_category_pct.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # Set tick labels
        ax.set_xticks(range(len(country_category_pct.columns)))
        ax.set_xticklabels(country_category_pct.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(country_category_pct.index)))
        ax.set_yticklabels(country_category_pct.index)
        
        # Add value labels
        for i in range(len(country_category_pct.index)):
            for j in range(len(country_category_pct.columns)):
                value = country_category_pct.iloc[i, j]
                if value > 0:
                    text_color = 'white' if value > 50 else 'black'
                    text = ax.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center", color=text_color, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
        
        ax.set_title('Multi-Country News Category Analysis Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        return fig
    
    def generate_time_series(self, df):
        """
        Generate news time series chart
        
        Args:
        df (DataFrame): DataFrame containing news data
        
        Returns:
        fig: Matplotlib figure object
        """
        # Convert date format
        try:
            df['date_parsed'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%SZ', errors='coerce')
            df = df.dropna(subset=['date_parsed'])
            
            # Group by date and count
            daily_counts = df.groupby(df['date_parsed'].dt.date).size()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            daily_counts.plot(kind='line', marker='o', ax=ax, color='purple')
            
            # Add labels and title
            ax.set_title('News Publishing Time Trend', fontsize=15)
            ax.set_xlabel('Date')
            ax.set_ylabel('News Count')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Failed to generate time series chart: {str(e)}")
            # Return empty chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'Unable to generate time series chart', ha='center', va='center')
            ax.axis('off')
            return fig

# Usage example
if __name__ == "__main__":
    # Create sample data
    data = {
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'positive'],
        'sourcecountry': ['USA', 'China', 'UK', 'USA', 'India'],
        'keywords': [
            ['AI', 'technology'], 
            ['climate', 'crisis'], 
            ['economy', 'growth'],
            ['health', 'vaccine'],
            ['education', 'online']
        ],
        'categories': [
            ['Technology', 'Innovation'], 
            ['Environment', 'Politics'], 
            ['Business', 'Economy'],
            ['Health', 'Science'],
            ['Education', 'Online']
        ],
        'date': ['20231201T120000Z', '20231202T130000Z', '20231203T140000Z', '20231204T150000Z', '20231205T160000Z']
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create visualizer instance
    visualizer = NewsVisualizer()
    
    # Generate charts
    fig1 = visualizer.generate_sentiment_chart(df)
    fig2 = visualizer.generate_country_chart(df)
    fig3 = visualizer.generate_category_chart(df)
    
    # Show charts
    plt.show() 
