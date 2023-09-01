import pandas as pd # data processing library 
import matplotlib.pyplot as plt # for plotting data findings
from textblob import TextBlob # for sentiment analysis
from googletrans import Translator # for translation
import traceback # for exception handling
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_excel('C:/Users/faaiz/Downloads/Hello/AI_Engineer_Dataset_Task_1.xlsx')

# bringing only comments out into the dataframe
user_comments_df = df[df['QuestionType'] == 'User Comment']

translator = Translator()
lemmatizer = WordNetLemmatizer()
# custom function for sentiment analysis
def get_sentiment(text):
    if isinstance(text, str):  # check if text string.
        try:
            words = word_tokenize(text)
            lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
            text = ' '.join(lemmatized_words)
            # Translate the text to English
            
            #translated_text = translator.translate(text, src='ar', dest='en').text
            analysis = TextBlob(text)  # change text to translated_text if uncommented translation code
            if analysis.sentiment.polarity > 0:
                return 'Positive'
            elif analysis.sentiment.polarity < 0:
                return 'Negative'
            else:
                return 'Neutral'
        except Exception as e:
            pass
    else:
        return 'NaN'  # arabic comments and Null values being returned as NaN 

# applying sentiment analysis and placing it under sentiment column 
user_comments_df['Sentiment'] = user_comments_df['ParticipantResponse'].apply(get_sentiment)

# data visualization
sentiment_counts = user_comments_df['Sentiment'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution of User Comments')
plt.show()


