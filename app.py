import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

nltk.download('stopwords')
set(stopwords.words('english'))

# Create feedback directory if it doesn't exist
feedback_dir = 'feedback'
os.makedirs(feedback_dir, exist_ok=True)
combined_file_path = os.path.join(feedback_dir, 'combined_feedback.csv')

# Define function for sentiment analysis
def analyze_sentiment(text):
    stop_words = stopwords.words('english')
    text_final = ''.join(c for c in text if not c.isdigit())
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound']) / 2, 2)

    sentiment = "Neutral"
    if compound > 0.6:
        sentiment = "Positive"
    elif compound < 0.4:
        sentiment = "Negative"

    return processed_doc1, dd, compound, sentiment

# Streamlit UI
st.title('Sentiment Analysis with Streamlit')

text_input = st.text_area("Enter text for sentiment analysis", height=150)

if st.button('Analyze'):
    processed_text, dd, compound, sentiment = analyze_sentiment(text_input)

    # Display results
    st.subheader('Sentiment Analysis Result')
    st.write(f"**Text:** {processed_text}")
    st.write(f"**Positive Score:** {dd['pos']}")
    st.write(f"**Neutral Score:** {dd['neu']}")
    st.write(f"**Negative Score:** {dd['neg']}")
    st.write(f"**Compound Score:** {compound}")
    st.write(f"**Sentiment:** **{sentiment}**")

    # Save feedback
    feedback = {
        'text': processed_text,
        'positive': dd['pos'],
        'negative': dd['neg'],
        'neutral': dd['neu'],
        'compound': compound,
        'sentiment': sentiment
    }

    feedback_file_path = os.path.join(feedback_dir, f"{sentiment}_{len(os.listdir(feedback_dir))}.csv")
    pd.DataFrame([feedback]).to_csv(feedback_file_path, index=False)

    if os.path.exists(combined_file_path):
        combined_df = pd.read_csv(combined_file_path)
    else:
        combined_df = pd.DataFrame()

    combined_df = pd.concat([combined_df, pd.DataFrame([feedback])], ignore_index=True)
    combined_df.to_csv(combined_file_path, index=False)

    # Plot the sentiment scores
    st.subheader('Sentiment Analysis Scores')
    fig, ax = plt.subplots()
    categories = ['Positive', 'Negative', 'Neutral']
    values = [dd['pos'], dd['neg'], dd['neu']]
    ax.bar(categories, values, color=['green', 'red', 'blue'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Scores')
    ax.set_title('Sentiment Analysis Scores')
    st.pyplot(fig)

# Display combined feedback file for download
if os.path.exists(combined_file_path):
    st.subheader('Download Combined Feedback')
    with open(combined_file_path, 'rb') as f:
        st.download_button('Download Combined Feedback', f, file_name='combined_feedback.csv')
