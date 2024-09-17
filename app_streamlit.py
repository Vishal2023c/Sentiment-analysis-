import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import os
import matplotlib.pyplot as plt
import speech_recognition as sr
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Create feedback directory if it doesn't exist
feedback_dir = 'feedback'
os.makedirs(feedback_dir, exist_ok=True)
combined_file_path = os.path.join(feedback_dir, 'combined_feedback.csv')

# Define function for sentiment analysis with enhanced negation handling
def analyze_sentiment(text):
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text)

    # Adjust compound score for negation handling
    compound = dd['compound']

    # List of negation words and their combinations
    negations = ['not', 'no', 'never', 'none', 'neither', 'nor',
                "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't",
                "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't",
                "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
                "ain't", "cannot", "can't", "daren't", "didn't", "hadn't",
                "hasn't", "haven't", "isn't", "mightn't", "mustn't", "shan't",
                "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]

    # Additional combinations with negations and sentiment words
    combinations = [
        ['not', 'good'],        # e.g., not good
        ['not', 'bad'],         # e.g., not bad
        ['no', 'good'],         # e.g., no good
        ['no', 'bad'],          # e.g., no bad
        ['never', 'good'],      # e.g., never good
        ['never', 'bad'],       # e.g., never bad
        ['none', 'good'],       # e.g., none good
        ['none', 'bad'],        # e.g., none bad
        ['neither', 'good'],    # e.g., neither good
        ['neither', 'bad'],     # e.g., neither bad
        ['nor', 'good'],        # e.g., nor good
        ['nor', 'bad'],         # e.g., nor bad
        ["couldn't", 'good'],   # e.g., couldn't good
        ["couldn't", 'bad'],    # e.g., couldn't bad
        ["shouldn't", 'good'],  # e.g., shouldn't good
        ["shouldn't", 'bad'],   # e.g., shouldn't bad
        ["wouldn't", 'good'],   # e.g., wouldn't good
        ["wouldn't", 'bad'],    # e.g., wouldn't bad
        ["didn't", 'good'],     # e.g., didn't good
        ["didn't", 'bad'],      # e.g., didn't bad
        ["doesn't", 'good'],    # e.g., doesn't good
        ["doesn't", 'bad'],     # e.g., doesn't bad
        ["don't", 'good'],      # e.g., don't good
        ["don't", 'bad'],       # e.g., don't bad
        ["hadn't", 'good'],     # e.g., hadn't good
        ["hadn't", 'bad'],      # e.g., hadn't bad
        ["hasn't", 'good'],     # e.g., hasn't good
        ["hasn't", 'bad'],      # e.g., hasn't bad
        ["haven't", 'good'],    # e.g., haven't good
        ["haven't", 'bad'],     # e.g., haven't bad
        ["isn't", 'good'],      # e.g., isn't good
        ["isn't", 'bad'],       # e.g., isn't bad
        ["mightn't", 'good'],   # e.g., mightn't good
        ["mightn't", 'bad'],    # e.g., mightn't bad
        ["mustn't", 'good'],    # e.g., mustn't good
        ["mustn't", 'bad'],     # e.g., mustn't bad
        ["needn't", 'good'],    # e.g., needn't good
        ["needn't", 'bad'],     # e.g., needn't bad
        ["shan't", 'good'],     # e.g., shan't good
        ["shan't", 'bad'],      # e.g., shan't bad
        ["wasn't", 'good'],     # e.g., wasn't good
        ["wasn't", 'bad'],      # e.g., wasn't bad
        ["weren't", 'good'],    # e.g., weren't good
        ["weren't", 'bad'],     # e.g., weren't bad
        ["won't", 'good'],      # e.g., won't good
        ["won't", 'bad'],       # e.g., won't bad
        ["ain't", 'good'],      # e.g., ain't good
        ["ain't", 'bad'],       # e.g., ain't bad
        ["cannot", 'good'],     # e.g., cannot good
        ["cannot", 'bad'],      # e.g., cannot bad
        ["can't", 'good'],      # e.g., can't good
        ["can't", 'bad'],       # e.g., can't bad
        ["daren't", 'good'],    # e.g., daren't good
        ["daren't", 'bad'],     # e.g., daren't bad
        ["didn't", 'good'],     # e.g., didn't good
        ["didn't", 'bad'],      # e.g., didn't bad
        ["hadn't", 'good'],     # e.g., hadn't good
        ["hadn't", 'bad'],      # e.g., hadn't bad
        ["hasn't", 'good'],     # e.g., hasn't good
        ["hasn't", 'bad'],      # e.g., hasn't bad
        ["haven't", 'good'],    # e.g., haven't good
        ["haven't", 'bad'],     # e.g., haven't bad
        ["isn't", 'good'],      # e.g., isn't good
        ["isn't", 'bad'],       # e.g., isn't bad
        ["mightn't", 'good'],   # e.g., mightn't good
        ["mightn't", 'bad'],    # e.g., mightn't bad
        ["mustn't", 'good'],    # e.g., mustn't good
        ["mustn't", 'bad'],     # e.g., mustn't bad
        ["needn't", 'good'],    # e.g., needn't good
        ["needn't", 'bad'],     # e.g., needn't bad
        ["shan't", 'good'],     # e.g., shan't good
        ["shan't", 'bad'],      # e.g., shan't bad
        ["shouldn't", 'good'],  # e.g., shouldn't good
        ["shouldn't", 'bad'],   # e.g., shouldn't bad
        ["wasn't", 'good'],     # e.g., wasn't good
        ["wasn't", 'bad'],      # e.g., wasn't bad
        ["weren't", 'good'],    # e.g., weren't good
        ["weren't", 'bad'],     # e.g., weren't bad
        ["won't", 'good'],      # e.g., won't good
        ["won't", 'bad'],       # e.g., won't bad
        ["wouldn't", 'good'],   # e.g., wouldn't good
        ["wouldn't", 'bad'],    # e.g., wouldn't bad
    ]


    words = text.split()

    for combo in combinations:
        if all(word in words for word in combo):
            if combo[0] in negations:
                idx = words.index(combo[0])
                next_word = words[idx + 1].lower()
                if next_word in sa.lexicon:
                    compound *= -1.0  # Invert compound score for negated word

    # Round compound score and determine sentiment
    compound = round((1 + compound) / 2, 2)

    # Determine sentiment category
    sentiment = "Neutral"
    if compound > 0.6:
        sentiment = "Positive"
    elif compound < 0.4:
        sentiment = "Negative"

    return text, dd, compound, sentiment

# Define function to calculate accuracy and negative label form out of 100
def calculate_accuracy(sentiment):
    if sentiment == "Positive":
        accuracy = dd['pos'] * 100
        label = "Positive"
    elif sentiment == "Negative":
        accuracy = dd['neg'] * 100
        label = "Negative"
    else:
        accuracy = dd['neu'] * 100
        label = "Neutral"

    return accuracy, label

# Define function to fetch feedback data
def load_feedback():
    if os.path.exists(combined_file_path):
        combined_df = pd.read_csv(combined_file_path)
        pos_df = combined_df[combined_df['sentiment'] == 'Positive'] * 100
        neu_df = combined_df[combined_df['sentiment'] == 'Neutral'] * 100
        neg_df = combined_df[combined_df['sentiment'] == 'Negative'] * 100
        return pos_df, neu_df, neg_df, combined_df
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Function to handle voice commands
def handle_voice_command(command):
    pos_df, neu_df, neg_df, combined_df = load_feedback()

    if 'show last' in command:
        try:
            num = int(command.split()[-1])
        except ValueError:
            num = 5  # Default to show last 5 if no specific number is mentioned
        st.subheader(f"Last {num} Feedback Entries")
        st.write(combined_df.tail(num))

    elif 'show first' in command or 'first entries' in command:
        try:
            num = int(command.split()[-1])
        except ValueError:
            num = 5  # Default to show first 5 if no specific number is mentioned
        st.subheader(f"First {num} Feedback Entries")
        st.write(combined_df.head(num))

    elif 'show +ve' in command or 'show positive' in command:
        st.subheader("Positive Feedback Entries")
        st.write(pos_df)

    elif 'show -ve' in command or 'show negative' in command:
        st.subheader("Negative Feedback Entries")
        st.write(neg_df)

    elif 'show neutral' in command:
        st.subheader("Neutral Feedback Entries")
        st.write(neu_df)

    elif 'count +ve' in command or 'count positive' in command:
        st.subheader("Count of Positive Feedback Entries")
        st.write(len(pos_df))

    elif 'count -ve' in command or 'count negative' in command:
        st.subheader("Count of Negative Feedback Entries")
        st.write(len(neg_df))

    elif 'count neutral' in command:
        st.subheader("Count of Neutral Feedback Entries")
        st.write(len(neu_df))

    elif 'show all' in command:
        st.subheader("All Feedback Entries")
        st.write(combined_df)

    else:
        st.warning("Command not recognized. Please try again.")

# Define function to capture voice input
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    st.info("Processing voice command...")
    try:
        command = recognizer.recognize_google(audio)
        st.success(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        st.error("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        st.error("Sorry, the speech service is unavailable.")
        return ""

# Streamlit UI
st.title('Sentiment Analysis with Streamlit')

# Sentiment Analysis Section
st.header('Sentiment Analysis')
text_input = st.text_area("Enter text for sentiment analysis", height=150)

if st.button('Analyze'):
    processed_text, dd, compound, sentiment = analyze_sentiment(text_input)

    # Calculate accuracy and label
    accuracy, label = calculate_accuracy(sentiment)

    # Display results
    st.subheader('Sentiment Analysis Result')
    st.write(f"**Text:** {processed_text}")
    st.write(f"**Positive Score:** {dd['pos']}")
    st.write(f"**Neutral Score:** {dd['neu']}")
    st.write(f"**Negative Score:** {dd['neg']}")
    st.write(f"**Compound Score:** {compound}")
    st.write(f"**Sentiment:** **{sentiment}**")

    # Display accuracy and negative label form out of 100
    st.subheader('Accuracy and Negative Label Form')
    st.write(f"**Accuracy for {label} Sentiment:** {accuracy}%")
    if sentiment == "Negative":
        st.write(f"**Negative Label Form:** {100 - accuracy}%")

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

# Feedback Data Section
st.header('Fetch Feedback Data')
st.write("Click the microphone icon and speak your command (e.g., 'show last 5', 'show first 5', 'show positive', 'show negative', 'show neutral', 'count positive', 'count negative', 'count neutral').")

if st.button("🎙️ Speak Command"):
    command = recognize_speech_from_mic()
    if command:
        handle_voice_command(command)

# Display combined feedback file for download
if os.path.exists(combined_file_path):
    st.subheader('Download Combined Feedback')
    with open(combined_file_path, 'rb') as f:
        st.download_button('Download Combined Feedback', f, file_name='combined_feedback.csv')
