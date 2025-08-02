import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Import custom modules
from news_analyzer import NewsAnalyzer
from db_manager import DBManager
from visualizer import NewsVisualizer

# Load environment variables
load_dotenv()


# Initialize components
news_analyzer = NewsAnalyzer("machine_learning")
db_manager = DBManager()
visualizer = NewsVisualizer()

# Set page title
st.set_page_config(
    page_title="News Sentiment and Diversity Analysis Across Different Regions",
    page_icon="📰",
    layout="wide"
)

# Main application
def main():
    st.title("📰 News Sentiment and Diversity Analysis Across Different Regions")
    st.markdown("---")
    
    # Initialize session_state
    if 'show_clear_confirm' not in st.session_state:
        st.session_state.show_clear_confirm = False
    
    # Sidebar - Navigation menu
    with st.sidebar:
        st.header("Search News")
        keyword = st.text_input("Enter keyword (optional)", "")
        # MY SN UK FR CA AS
        country = st.selectbox("Select country", ["ALL", "CH", 'MY', 'AU', 'NZ', 'UK', 'FR', 'CA', "US", 'NG','ZA'], index=0, key="search_country_selector")
        timespan = st.selectbox("Time range", ["1d", "3d", "7d", "14d", "30d"], index=2)
        max_records = st.slider("Number of news articles", 10, 250, 20)
        
        search_btn = st.button("Search and Analyze", type="primary")
        
        page = st.radio("Select function", ["Batch News Analysis", "Single News Analysis"])
        
        # Data management part
        st.markdown("### Data Management")
        
        # Show current number of news in database
        current_news_count = db_manager.get_news_count()        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            if current_news_count > 0:
                # Use session_state to manage confirmation status
                st.session_state.show_clear_confirm = True
            else:
                st.warning("No data in database to clear")
        
        # Show confirmation dialog
        if st.session_state.get('show_clear_confirm', False):
            st.warning("⚠️ Are you sure you want to clear all data? This action cannot be undone!")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Confirm Clear", type="primary"):
                    with st.spinner("Clearing data..."):
                        if db_manager.clear_all_news():
                            st.success("✨ Data cleared successfully!")
                            st.session_state.show_clear_confirm = False
                            st.rerun()
                        else:
                            st.error("Failed to clear data, please try again")
            
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.show_clear_confirm = False
                    st.rerun()
    
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            """
            <div>
            📋 <strong><a href='https://docs.google.com/document/d/1hcbd4DGGXUmwcRO8cySfuAhRodaJQv5ZhdR1zKDDqd0/edit?usp=sharing' target='_blank'>View User Manual</a></strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("This application retrieves global news through GDELT API, analyzes news content using machine learning algorithms, and provides data visualization.")
    
    # Display different content based on selected page
    if page == "Batch News Analysis":
        batch_news_analysis(search_btn, keyword, country, timespan, max_records)
    else:
        single_news_analysis()

def batch_news_analysis(search_btn=None, keyword=None, country=None, timespan=None, max_records=None):
    """Batch news analysis page"""
    st.header("News Analysis and Data Visualization")
    
    # Show model information (always displayed)
    model_info = news_analyzer.get_model_info()
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Sentiment Analysis Model:**\n{model_info['sentiment_model']}")
    with col2:
        st.info(f"**Classification Model:**\n{model_info['classification_model']}")
    
    # Main interface
    if search_btn:
        search_and_analyze_news(keyword, country, timespan, max_records)
    
    # Show data and visualizations
    display_visualizations()

def single_news_analysis():
    """Single news analysis page"""
    st.header("Single News Analysis")
    st.markdown("Enter news URL to analyze news content directly without saving to database.")
    
    # Input URL
    demo_url = 'https://www.thenews.com.pk/latest/1308481-uae-to-introduce-ai-education-across-all-govt-schools'
    news_url = st.text_input("Enter news URL", demo_url)
    news_title = st.text_input("Enter news title (optional)", "")
    
    # Analyze button
    if st.button("Analyze News", type="primary"):
        if not news_url or news_url == "https://example.com/news":
            st.error("Please enter a valid news URL")
        else:
            with st.spinner("Analyzing news..."):
                if not news_title:
                    news_title = "Unknown Title"
                
                # Print analyzer
                print("Analyzer type: ", news_analyzer.analyzer)
                news_analyzer.set_analyzer("machine_learning")
                # Analyze news
                analysis = news_analyzer.analyze_content(
                    news_url,
                    news_title,
                    "",  # Date is empty
                    ""   # Source country is empty
                )
                
                if analysis:
                    # Show analysis results
                    st.success("News analysis completed!")
                    
                    st.subheader("News Basic Information")
                    st.markdown(f"**Title:** {analysis['title']}")
                    st.markdown(f"**URL:** [{analysis['url']}]({analysis['url']})")
                    if analysis['date']:
                        st.markdown(f"**Date:** {analysis['date']}")
                    if analysis['sourcecountry']:
                        st.markdown(f"**Source Country:** {analysis['sourcecountry']}")
                
                    st.subheader("Sentiment Analysis")
                    sentiment = analysis['sentiment']
                    # Show different colors and icons based on sentiment
                    if sentiment == "positive":
                        st.markdown("😊 **Positive**", unsafe_allow_html=True)
                        st.progress(0.8)
                    elif sentiment == "negative":
                        st.markdown("😞 **Negative**", unsafe_allow_html=True)
                        st.progress(0.2)
                    else:
                        st.markdown("😐 **Neutral**", unsafe_allow_html=True)
                        st.progress(0.5)
                    
                    # Show keywords and categories
                    st.subheader("Keywords")
                    if analysis['keywords']:
                        keywords_html = ' '.join([f'<span style="background-color: #e6f3ff; padding: 3px 8px; margin: 2px; border-radius: 15px; display: inline-block;">{k}</span>' for k in analysis['keywords']])
                        st.markdown(f"<div style='line-height: 2.5;'>{keywords_html}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No keywords detected")
                    
                    st.subheader("News Categories")
                    if analysis['categories']:
                        categories_html = ' '.join([f'<span style="background-color: #f0f0f0; padding: 3px 8px; margin: 2px; border-radius: 15px; display: inline-block;">{c}</span>' for c in analysis['categories']])
                        st.markdown(f"<div style='line-height: 2.5;'>{categories_html}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No categories detected")
                    
                    # WordCloud part has been removed because the WordCloud module has problems
                else:
                    st.error("News analysis failed, please check URL or try again later")

def search_and_analyze_news(keyword, country, timespan, max_records):
    """Search and analyze news"""
    countries = [country]
    if country == "ALL":
        countries = ["CH", 'MY', 'AU', 'NZ', 'UK', 'FR', 'CA', "US", 'NG','ZA']
    
    # Create a placeholder that can be cleared
    status_text = st.empty()
    fetch_text = st.empty()
    progress_bar = st.empty()
    prediction_results = st.empty()
    
    total_analyzed = 0
    
    for country_code, i in zip(countries, range(len(countries))):
        with st.spinner(f"Country count({i+1}/{len(countries)}), fetching ({country_code}) news..."):
            try:
                news_data = news_analyzer.fetch_news(keyword, country_code, timespan, max_records)
                
                if news_data is None:
                    st.error(f"The keyword did not obtain the news data of the {country_code}, Please change the keyword and search again.")
                    continue
                
                if not isinstance(news_data, dict) or "articles" not in news_data:
                    st.error(f"❌ Invalid response format for {country_code}. Expected JSON with 'articles' field.")
                    continue
                
                # Data validation passed, continue processing
                if len(news_data["articles"]) == 0:
                    st.warning(f"⚠️ No articles found for {country_code}")
                    continue
                
                # Successfully fetched data, start processing
                fetch_text.success(f"Successfully fetched {len(news_data['articles'])} news articles")
                
                # Create progress bar
                progress_bar_widget = progress_bar.progress(0)
                
                # Limit the number of news to analyze, avoid too many API calls
                max_to_analyze = len(news_data["articles"])
                
                for j, article in enumerate(news_data["articles"][:max_to_analyze]):
                    status_text.text(f"🔍 Analyzing article {j+1}/{max_to_analyze}: {article['title'][:50]}...")
                    
                    # If the news exists in the database, skip
                    if db_manager.check_news_exists(article["url"]):
                        print(f"Article {j+1} already exists, skipping")
                        continue
                    
                    # Analyze news content
                    analysis = news_analyzer.analyze_content(
                        article["url"],
                        article["title"],
                        article.get("seendate", ""),
                        article.get("sourcecountry", "")
                    )
                    if analysis is None:
                        print(f"Article {j+1} content fetch failed, URL: {article['url']}")
                        continue
                    
                    # Save to database
                    if analysis:
                        analysis['url'] = article["url"]
                        db_manager.save_news(analysis)
                        total_analyzed += 1
                        
                        # Show prediction results in real time
                        with prediction_results.container():
                            st.markdown("### 🎯 Latest Prediction Result")
                            
                            # Show news basic information
                            st.markdown(f"**Title:** {analysis['title'][:100]}...")
                            st.markdown(f"**URL:** [{analysis['url']}]({analysis['url']})")
                            st.markdown(f"**Source Country:** {analysis['sourcecountry']}")
                            
                            # Show prediction results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sentiment = analysis['sentiment']
                                sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                                st.markdown(f"**Sentiment:** {sentiment_emoji.get(sentiment, '😐')} {sentiment.title()}")
                            
                            with col2:
                                if analysis['categories']:
                                    categories_str = ', '.join(analysis['categories'][:2])  # Show first 2 categories
                                    st.markdown(f"**Categories:** {categories_str}")
                                else:
                                    st.markdown("**Categories:** None")
                            
                            with col3:
                                if analysis['keywords']:
                                    keywords_str = ', '.join(analysis['keywords'][:3])  # Show first 3 keywords
                                    st.markdown(f"**Keywords:** {keywords_str}")
                                else:
                                    st.markdown("**Keywords:** None")
                            
                            st.markdown("---")
                    
                    # Update progress bar
                    progress_bar_widget.progress((j + 1) / max_to_analyze)
                
                status_text.text("✅ Analysis completed!")
                fetch_text.success(f"Successfully analyzed and stored {max_to_analyze} news articles")
                
            except Exception as e:
                st.error(f"❌ Unexpected error fetching news for {country_code}: {str(e)}")
                continue
    
    # Clear all temporary display information
    status_text.empty()
    fetch_text.empty()
    progress_bar.empty()
    prediction_results.empty()
    
    # Show final completion dialog
    if total_analyzed > 0:
        st.balloons()  # Add celebration animation
        st.success(f"""
        🎉 **Data Processing Completed Successfully!**
        
        **Total Analyzed:** {total_analyzed} articles
        
        ### 📋 Next Steps - View Your Analysis Results:
        
        **1. 📊 Start with "News Data" tab** - View the complete data table and basic statistics
        
        **2. 😊 Check "Sentiment Analysis" tab** - See emotional tendencies across different regions
        
        **3. 📋 Explore "Category Analysis" tab** - Discover news topic diversity by country
        
        **4. ☁️ Review "Word Cloud" tab** - Visualize popular keywords and themes
        
        **💡 Pro Tips:**
        - Use the **country selector** above the tabs to compare specific countries
        - Select "All" to see cross-country comparisons and insights
        - Each tab provides both visual charts and detailed data tables
        """)
    else:
        st.success(f"""
        🎉 **Data Processing Completed!**
        
        ### 📋 Next Steps - View Your Analysis Results:
        
        **1. 📊 Start with "News Data" tab** - View the complete data table and basic statistics
        
        **2. 😊 Check "Sentiment Analysis" tab** - See emotional tendencies across different regions  
        
        **3. 📋 Explore "Category Analysis" tab** - Discover news topic diversity by country
        
        **4. ☁️ Review "Word Cloud" tab** - Visualize popular keywords and themes
        
        **💡 Tips:**
        - Use the **country selector** above the tabs to focus on specific regions
        - Each tab provides comprehensive analysis and visualizations
        """)

def display_visualizations():
    """Display data visualizations"""
    # Get all news data
    df = db_manager.get_all_news()
    
    if not df.empty:        
        # Add global country selector
        countries = df['sourcecountry'].unique()
        countries = sorted([country for country in countries if country and country.strip()])
        
        if countries:
            # Add "All" option
            country_options = ["All"] + countries
            selected_country = st.selectbox("Select Country (view only selected countries)", country_options, key="global_country_selector")
            
            # Filter data based on selected country
            if selected_country != "All":
                filtered_df = df[df['sourcecountry'] == selected_country]
                st.info(f"Showing {len(filtered_df)} news articles from {selected_country}")
            else:
                filtered_df = df
                st.info(f"Showing {len(filtered_df)} news articles from 10 countries across 5 continents. Click the button below to view tables and charts showing news sentiment and diversity analysis by region")
        else:
            filtered_df = df
            st.info("No valid country data, showing all news")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 News Data", 
            "😊 Sentiment Analysis", 
            "📋 Category Analysis",
            "☁️ Word Cloud"
        ])
        
        with tab1:
            st.subheader("News Data & Country Distribution Analysis")
            
            # Show simplified data table
            st.markdown("### News Data Table")
            display_df = filtered_df[["title", "sourcecountry", "categories", "sentiment", "url"]].copy()
            display_df.columns = ["Title", "Source Country", "Categories", "Sentiment", "URL"]
            
            # Configure URL column as clickable link
            st.dataframe(
                display_df, 
                use_container_width=True,
                column_config={
                    "URL": st.column_config.LinkColumn(
                        "URL",
                        help="Click to view original news",
                        validate="^https?://.*",
                        max_chars=100,
                        display_text="View Original"
                    )
                }
            )
            
            # Show basic data statistics
            st.markdown("### Basic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total News", len(filtered_df))
            with col2:
                st.metric("Source Countries", filtered_df["sourcecountry"].nunique())
            with col3:
                if len(filtered_df) > 0:
                    positive_count = len(filtered_df[filtered_df["sentiment"] == "positive"])
                    st.metric("Positive News Ratio", f"{positive_count/len(filtered_df):.1%}")
                else:
                    st.metric("Positive News Ratio", "0%")
            
            # Country distribution analysis part
            st.markdown("---")
            
            # Automatically determine display mode based on selected country
            if selected_country == "All":
                # Show distribution of all countries
                if len(df) > 0:
                    st.markdown("### News Source Country Distribution")
                    st.markdown("This analysis shows the distribution of news count and basic statistics by country.")
                    
                    # Generate country distribution bar chart
                    fig_country = visualizer.generate_country_chart(df, top_n=len(df['sourcecountry'].unique()))
                    st.pyplot(fig_country)
                    
                    # Show country statistics information
                    st.markdown("### Country Statistics Details")
                    country_stats = df['sourcecountry'].value_counts().reset_index()
                    country_stats.columns = ['Country', 'News Count']
                    country_stats['Percentage'] = (country_stats['News Count'] / country_stats['News Count'].sum() * 100).round(2)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(country_stats, use_container_width=True)
                    
                    with col2:
                        # Show statistical indicators
                        total_countries = len(country_stats)
                        total_news = country_stats['News Count'].sum()
                        avg_news_per_country = total_news / total_countries
                        top_country = country_stats.iloc[0]
                        
                        st.metric("Total Countries", total_countries)
                        st.metric("Total News", total_news)
                        st.metric("Avg News per Country", f"{avg_news_per_country:.1f}")
                        st.metric("Top Country", f"{top_country['Country']} ({top_country['News Count']} articles)")
                else:
                    st.info("Insufficient data for country distribution analysis")
            else:
                # Show detailed information of selected country
                if len(filtered_df) > 0:
                    st.markdown(f"### {selected_country} Country Details")
                    
                    # Rank of this country in all countries
                    country_stats = df['sourcecountry'].value_counts()
                    country_rank = list(country_stats.index).index(selected_country) + 1 if selected_country in country_stats.index else "Unknown"
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Country News Count", len(filtered_df))
                    with col2:
                        total_news = len(df)
                        percentage = (len(filtered_df) / total_news * 100) if total_news > 0 else 0
                        st.metric("Percentage of Total", f"{percentage:.1f}%")
                    with col3:
                        st.metric("Country Ranking", f"#{country_rank}" if country_rank != "Unknown" else "Unknown")
                    with col4:
                        avg_sentiment = filtered_df['sentiment'].value_counts()
                        most_common_sentiment = avg_sentiment.index[0] if len(avg_sentiment) > 0 else "Unknown"
                        sentiment_map = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
                        st.metric("Main Sentiment", sentiment_map.get(most_common_sentiment, most_common_sentiment))
                    
                    # Show comparison chart of this country with other countries
                    st.markdown("#### Comparison with Other Countries")
                    st.markdown("Shows news count distribution for all countries, highlighting the currently selected country.")
                    
                    # Generate distribution chart of selected country
                    fig_country = visualizer.generate_country_chart(df, top_n=len(df['sourcecountry'].unique()))
                    st.pyplot(fig_country)
                    
                    # Show category and sentiment distribution of this country
                    st.markdown("#### This Country's News Characteristics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sentiment Distribution**")
                        sentiment_dist = filtered_df['sentiment'].value_counts().reset_index()
                        sentiment_dist.columns = ['Sentiment', 'Count']
                        sentiment_dist['Sentiment'] = sentiment_dist['Sentiment'].map({
                            'positive': 'Positive', 
                            'negative': 'Negative', 
                            'neutral': 'Neutral'
                        })
                        st.dataframe(sentiment_dist, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Popular Categories**")
                        # Expand category data
                        all_categories = [cat for cats in filtered_df['categories'] for cat in cats]
                        if all_categories:
                            from collections import Counter
                            category_counter = Counter(all_categories)
                            top_categories = pd.DataFrame(category_counter.most_common(5), columns=['Category', 'Count'])
                            st.dataframe(top_categories, use_container_width=True)
                        else:
                            st.info("No category data")
                else:
                    st.info(f"No news data from {selected_country}")
        
        with tab2:
            st.subheader("Sentiment Analysis")
            
            # Automatically determine analysis mode based on selected country
            if selected_country == "All":
                # Multi-country comparison mode
                if len(df) > 0:  # Use all data
                    country_count = df['sourcecountry'].nunique()
                    if country_count >= 2:
                        st.markdown("### Multi-Country Sentiment Analysis Comparison")
                        st.markdown("This analysis shows differences in sentiment tendencies across countries, helping understand regional news characteristics.")
                        
                        # Add sentiment analysis legend
                        st.markdown("#### Sentiment Analysis Legend")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("**😊 Positive**")
                            st.markdown("- Optimistic, good news")
                            st.markdown("- Economic growth, achievements")
                            st.markdown("- Technological breakthroughs")
                        with col2:
                            st.markdown("**😐 Neutral**")
                            st.markdown("- Factual reporting")
                            st.markdown("- Balanced viewpoints")
                            st.markdown("- Objective information")
                        with col3:
                            st.markdown("**😞 Negative**")
                            st.markdown("- Bad news, problems")
                            st.markdown("- Conflicts, disasters")
                            st.markdown("- Economic decline")
                        
                        st.markdown("---")
                        
                        st.markdown("#### Grouped Bar Chart Comparison")
                        st.markdown("Shows distribution of positive, negative, and neutral sentiment for each country with absolute counts.")
                        
                        # Generate multi-country comparison chart
                        fig_comparison = visualizer.generate_sentiment_comparison_chart(df)
                        st.pyplot(fig_comparison)
                        
                        st.markdown("#### Heatmap Comparison")
                        st.markdown("This heatmap shows sentiment distribution percentages by country using color intensity, where green indicates high positive sentiment ratio and red indicates high negative sentiment ratio.")
                        
                        # Generate heatmap
                        fig_heatmap = visualizer.generate_sentiment_heatmap(df)
                        st.pyplot(fig_heatmap)
                        
                        # Show detailed data table
                        st.markdown("#### Detailed Data Statistics")
                        country_sentiment_detailed = df.groupby(['sourcecountry', 'sentiment']).size().unstack(fill_value=0)
                        
                        # Ensure all three sentiment types exist
                        expected_sentiments = ['negative', 'neutral', 'positive']
                        for sentiment in expected_sentiments:
                            if sentiment not in country_sentiment_detailed.columns:
                                country_sentiment_detailed[sentiment] = 0
                        
                        # Reorder and rename columns
                        country_sentiment_detailed = country_sentiment_detailed[expected_sentiments]
                        country_sentiment_detailed.columns = ['Negative', 'Neutral', 'Positive']
                        
                        # Add total column
                        country_sentiment_detailed['Total'] = country_sentiment_detailed.sum(axis=1)
                        
                        # Calculate percentages
                        for col in ['Negative', 'Neutral', 'Positive']:
                            country_sentiment_detailed[f'{col} Ratio'] = (
                                country_sentiment_detailed[col] / country_sentiment_detailed['Total'] * 100
                            ).round(1)
                        
                        st.dataframe(country_sentiment_detailed, use_container_width=True)
                    else:
                        st.info(f"Data contains only {country_count} countries, need at least 2 countries for sentiment comparison analysis.")
                else:
                    st.info("Insufficient data for multi-country sentiment comparison analysis")
            
            else:
                # Single country analysis mode
                if len(filtered_df) > 0:
                    st.markdown(f"### {selected_country} Sentiment Analysis")
                    fig = visualizer.generate_sentiment_chart(filtered_df)
                    st.pyplot(fig)
                    
                    # Add sentiment distribution table
                    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    sentiment_counts['Ratio'] = sentiment_counts['Count'] / sentiment_counts['Count'].sum()
                    
                    # Translate sentiment labels
                    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map({
                        'positive': 'Positive', 
                        'negative': 'Negative', 
                        'neutral': 'Neutral'
                    })
                    
                    st.dataframe(sentiment_counts, use_container_width=True)
                else:
                    st.info("Insufficient data to generate sentiment analysis chart")
        
        with tab3:
            st.subheader("News Category Analysis")
            
            # Automatically determine analysis mode based on selected country
            if selected_country == "All":
                # Multi-country comparison mode
                if len(df) > 0:  # Use all data
                    country_count = df['sourcecountry'].nunique()
                    if country_count >= 2:
                        st.markdown("### Multi-Country Category Analysis Comparison")
                        st.markdown("This analysis shows differences in news category distribution across countries, helping understand regional news diversity characteristics.")
                        
                        # Add category analysis legend
                        st.markdown("#### News Category Legend")
                        st.markdown("""
                        **News categories represent different topics and themes:**
                        - **World**: International affairs, global events, foreign policy, international relations, global conflicts and cooperation
                        - **Technology**: Innovation, AI, digital transformation, tech industry, scientific breakthroughs, emerging technologies
                        - **Business**: Financial markets, economic indicators, corporate news, trade, market analysis, business developments
                        - **Sports**: Athletic events, competitions, sports news, player transfers, tournament results, sports industry
                        """)
                        
                        st.markdown("---")
                        
                        # Set number of categories to display to 10
                        category_count = 10
                        
                        st.markdown("#### Grouped Bar Chart Comparison")
                        st.markdown("Shows distribution of popular news categories for each country with absolute counts.")
                        
                        # Generate multi-country category comparison chart
                        fig_category_comparison = visualizer.generate_category_comparison_chart(df, top_n_categories=category_count)
                        st.pyplot(fig_category_comparison)
                        
                        st.markdown("#### Heatmap Comparison")
                        st.markdown("This heatmap shows each country's focus on different news categories, with darker colors indicating higher proportion of news in that category.")
                        
                        # Generate category heatmap
                        fig_category_heatmap = visualizer.generate_category_heatmap(df, top_n_categories=category_count)
                        st.pyplot(fig_category_heatmap)
                        
                        # Show detailed data table
                        st.markdown("#### Detailed Data Statistics")
                        
                        # Expand category data for statistics
                        expanded_data = []
                        for _, row in df.iterrows():
                            for category in row['categories']:
                                expanded_data.append({
                                    'sourcecountry': row['sourcecountry'],
                                    'category': category
                                })
                        
                        if expanded_data:
                            expanded_df = pd.DataFrame(expanded_data)
                            top_categories = expanded_df['category'].value_counts().head(category_count).index.tolist()
                            filtered_expanded = expanded_df[expanded_df['category'].isin(top_categories)]
                            
                            category_detail = filtered_expanded.groupby(['sourcecountry', 'category']).size().unstack(fill_value=0)
                            
                            # Add total column
                            category_detail['Total'] = category_detail.sum(axis=1)
                            
                            # Calculate percentage columns
                            for col in top_categories:
                                if col in category_detail.columns:
                                    category_detail[f'{col}_Ratio'] = (
                                        category_detail[col] / category_detail['Total'] * 100
                                    ).round(1)
                            
                            st.dataframe(category_detail, use_container_width=True)
                        else:
                            st.info("No category data available")
                        
                    else:
                        st.info(f"Data contains only {country_count} countries, need at least 2 countries for multi-country category comparison analysis.")
                else:
                    st.info("Insufficient data for multi-country category comparison analysis")
            
            else:
                # Single country analysis mode
                if len(filtered_df) > 0:
                    st.markdown(f"### {selected_country} Category Analysis")
                    fig = visualizer.generate_category_chart(filtered_df)
                    st.pyplot(fig)
                else:
                    st.info("Insufficient data to generate category analysis chart")
        
        with tab4:
            st.subheader("Keywords Word Cloud")
            
            # Automatically determine analysis mode based on selected country
            if selected_country == "All":
                # Multi-country comparison mode
                if len(df) > 0:  # Use all data
                    country_count = df['sourcecountry'].nunique()
                    if country_count >= 2:
                        st.markdown("### Multi-Country Keywords Word Cloud Comparison")
                        st.markdown("This analysis shows keyword differences across countries, helping understand different regional news focus areas.")
                        
                        # Add word cloud legend
                        st.markdown("#### Word Cloud Legend")
                        st.markdown("""
                        **Word clouds visualize the most important keywords in news content:**
                        - **Font Size**: Larger words appear more frequently in the news
                        - **Color**: Different colors help distinguish between words
                        - **Position**: Words are arranged randomly for better visualization
                        - **Content**: Keywords are extracted from news titles and content using AI analysis
                        - **Regional Differences**: Each country's word cloud shows their unique news focus areas
                        """)
                        
                        st.markdown("---")
                        
                        st.markdown("#### Multi-Country Word Cloud Comparison")
                        st.markdown(f"Shows word clouds for all {country_count} countries, displaying the most popular keywords for each country.")
                        
                        # Generate multi-country word cloud comparison
                        fig_wordcloud_comparison = visualizer.generate_wordcloud_comparison(df)
                        st.pyplot(fig_wordcloud_comparison)
                        
                        # Show keyword statistics
                        st.markdown("#### Popular Keywords Statistics by Country")
                        
                        # Calculate popular keywords for each country
                        country_keywords_stats = []
                        top_countries = df['sourcecountry'].value_counts().index.tolist()
                        
                        for country in top_countries:
                            country_df = df[df['sourcecountry'] == country]
                            keywords_list = country_df['keywords'].tolist()
                            all_keywords = [keyword for keywords in keywords_list for keyword in keywords if keywords]
                            
                            if all_keywords:
                                from collections import Counter
                                keyword_freq = Counter(all_keywords)
                                top_5_keywords = [kw for kw, _ in keyword_freq.most_common(5)]
                                country_keywords_stats.append({
                                    'Country': country,
                                    'News Count': len(country_df),
                                    'Popular Keywords': ', '.join(top_5_keywords)
                                })
                        
                        if country_keywords_stats:
                            keywords_stats_df = pd.DataFrame(country_keywords_stats)
                            st.dataframe(keywords_stats_df, use_container_width=True)
                        else:
                            st.info("No keywords data available")
                        
                    else:
                        st.info(f"Data contains only {country_count} countries, need at least 2 countries for word cloud comparison analysis.")
                else:
                    st.info("Insufficient data for multi-country word cloud comparison analysis")
            
            else:
                # Single country analysis mode
                if len(filtered_df) > 0:
                    st.markdown(f"### {selected_country} News Keywords Word Cloud")
                    fig = visualizer.generate_wordcloud_by_country(filtered_df, selected_country)
                    st.pyplot(fig)
                else:
                    st.info("Insufficient data to generate word cloud")
        
    else:
        st.info("Please search for news first to get data")

if __name__ == "__main__":
    main() 
