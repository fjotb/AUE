import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from googletrans import Translator
from gensim import corpora
from gensim.models import LdaModel
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from nltk.stem import WordNetLemmatizer

# download nltk stopwords for english (initialization)
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
arabic_stop_words = set([
    'و', 'في', 'من', 'على', 'عن', 'هذا', 'هذه', 'ذلك', 'هم', 'هما',
    'هما', 'هن', 'هو', 'هي', 'كل', 'كما', 'نحن', 'أنا', 'أنتم', 'أنت',
    # Add more Arabic stopwords as needed
])


# Load the English and Arabic stopwords
english_stop_words = set(stopwords.words('english'))

df = pd.read_excel('C:/Users/faaiz/Downloads/Hello/AI_Engineer_Dataset_Task_1.xlsx')

# Bringing only comments out into the dataframe
user_comments_df = df[df['QuestionType'] == 'User Comment']

translator = Translator()

# Custom function includes translate, tokenizing (into single words), and stopwords removal for all the user comments
def translate_tokenize_and_remove_stopwords(text):
    if isinstance(text, str):  # Check if 'text' is a string.
        try:
            # Uncomment below line of code to feed translation
            # translated_text = translator.translate(text, src='ar', dest='en').text
            # Using word_tokenize function from nltk
            tokens = word_tokenize(text)  # Change text to translated_text if the translation line is uncommented
            # Remove English and Arabic stopwords
            tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in english_stop_words and word.lower() not in arabic_stop_words]
            return tokens
        except Exception as e:
            pass
    else:
        return []

# Applying translation, tokenization, and stopwords removal, and placing tokenized comments under the 'TokenizedComments' column
user_comments_df['TokenizedComments'] = user_comments_df['ParticipantResponse'].apply(translate_tokenize_and_remove_stopwords)

documents = user_comments_df['TokenizedComments'].tolist()

dictionary = corpora.Dictionary(documents)

# Create a document-term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]

# Latent Dirichlet Allocation
lda_model = LdaModel(doc_term_matrix, num_topics=5, id2word=dictionary, passes=20)

# Increase num_topics and passes for accuracy but increased computing time
topics = lda_model.print_topics(num_words=5)

# Visualize the topics using pyLDAvis
vis_data = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.display(vis_data)

# Print the top keywords for each topic
topics = lda_model.print_topics(num_words=10)  # Adjust the number of keywords as needed

# Plot the top keywords for each topic
for i, topic in enumerate(topics):
    top_words = [val.split('*')[1].strip() for val in topic[1].split('+')]
    word_freq = [float(val.split('*')[0]) for val in topic[1].split('+')]

    plt.figure(figsize=(8, 6))
    plt.barh(top_words, word_freq)
    plt.gca().invert_yaxis()
    plt.xlabel('Word Frequency')
    plt.title(f'Topic {i + 1}')
    plt.show()
