import pandas as pd
from textblob import TextBlob

print("=" * 60)
print("        SENTIMENT ANALYSIS")
print("=" * 60)

# Load Dataset
df = pd.read_csv("reviews.csv")

# Function to Analyze Sentiment
def analyze_sentiment(review):
    analysis = TextBlob(str(review))
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Sentiment"] = df["Review"].apply(analyze_sentiment)

# Display Results
print("\nSentiment Analysis Results:\n")
print(df)

# Count Sentiments
print("\nSentiment Summary:\n")
print(df["Sentiment"].value_counts())

# Save Output
df.to_csv("sentiment_output.csv", index=False)

print("\nAnalysis completed successfully!")
print("Results saved as 'sentiment_output.csv'")
