import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_excel('C:/Users/faaiz/Downloads/Hello/AI_Engineer_Dataset_Task_1.xlsx')


relevant_question_types = ['Rating', 'User Comment']
filtered_df = df[df['QuestionType'].isin(relevant_question_types)]
lemmatizer = WordNetLemmatizer()
# custom function for sentiment analysis using 
def analyze_sentiment(text):
    try:
        if isinstance(text, float):  # Check if the value is a float (e.g., NaN)
            return 'Neutral'
        words = word_tokenize(str(text))
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
        text = ' '.join(lemmatized_words)
        #sentiment analysis using TextBlob library
        analysis = TextBlob(str(text))
        
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Error'  # Handle any exceptions during analysis

# applying sentiment analysis to each element in participant response column and adding
# sentiment under sentiment column
filtered_df['Sentiment'] = filtered_df['ParticipantResponse'].apply(analyze_sentiment)

print("Sentiment Analysis Results:")
print(filtered_df[['ParticipantResponse', 'Sentiment']])

# Form analysis by counting sentiments
sentiment_counts = filtered_df['Sentiment'].value_counts()
print("\nSentiment Statistics:")
print(sentiment_counts)
