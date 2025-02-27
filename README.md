# Sentiment-Negation-Analytics

This repository contains a Streamlit application for sentiment analysis enhanced with comprehensive negation handling. It allows users to analyze the sentiment of text inputs, visualize sentiment scores, manage feedback, and interact via voice commands.

## Getting Started

## Project Structure
```
Sentiment-Negation-Analytics/
│
├── .gitignore # Git ignore file
├── app.py # Core sentiment analysis functions
├── app_streamlit.py # Main Streamlit application script
├── dirFileStruct.txt # Directory file structure details
├── feedback/
│ ├── combined_feedback.csv # Combined feedback data file (generated)
│ └── individual_feedback.csv # Individual feedback files (generated)
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)
- Git (optional, for version control)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/CodeSage4D/Sentiment-Negation-Analytics.git
    cd Sentiment-Negation-Analytics
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```sh
    python -m venv env
    env\Scripts\activate      # On Windows use `env\Scripts\activate`
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit application:**

    ```sh
    streamlit run app_streamlit.py
    ```

2. **Interact with the application:**
    - Enter text in the provided input box for sentiment analysis.
    - Click on the "Analyze" button to see sentiment analysis results, including positive, neutral, and negative scores, and the overall sentiment classification.
    - View sentiment scores and visualizations.

### Voice Commands

- Use the voice command feature by clicking on the microphone icon in the application.
- Available commands include:
  - "show last 5"
  - "show positive"
  - "count negative"
  - "download combined feedback"

### Feedback Management

- Feedback data is stored in the `feedback/` directory.
- Individual feedback files are automatically generated and combined into `combined_feedback.csv`.

## Refrences of Model
- https://github.com/cjhutto/vaderSentiment
- https://pypi.org/project/vaderSentiment/
- Medium (https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664)
- VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text (https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=vanders)

