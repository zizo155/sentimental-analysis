import streamlit as st


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
import string
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
import nltk
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import TfidfVectorizer

# Title the app
st.title('Sentimental Analysis')


# Load data
df = pd.read_csv("sentimentdataset.csv")


# Set up the sidebar
st.sidebar.title('Developer Information')
st.sidebar.text('Welcome to my Streamlit app!')
# Display name at the top of the sidebar
st.sidebar.subheader('Developer:')
st.sidebar.text('Zohreh Taghibakhshi')
#link to GitHub profile
st.sidebar.markdown('[GitHub](https://github.com/zizo155/sentimental-analysis)')




# Display the head of the data
st.subheader('Raw Data - Head')
st.write(df.head())

# Display list of columns
st.subheader('List of Columns')
columns_table = pd.DataFrame(df.columns.tolist(), columns=["Columns"])
st.table(columns_table)


# --------------------------------------------
# data preprocessing
# Converts the 'Timestamp' column in DataFrame df to datetime format
#Extracts the day component from the 'Timestamp' column and assigns it to a new columns named 'Day',
# 'month' and 'year' in the DataFrame df

# drop unnecessary columns
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Hashtags','Day', 'Hour','Sentiment'])

#time stamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Day_of_Week'] = df['Timestamp'].dt.day_name()

# mapping
month_mapping = {
    1: 'Jnuary',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

df['Month'] = df['Month'].map(month_mapping)

df['Month'] = df['Month'].astype('object')


# cleaning texts
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = " ".join(text.split())
    tokens = word_tokenize(text)

    cleaned_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]

    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

df["Clean_Text"] = df["Text"].apply(clean)

# removes leading and trailing white spaces from below columns
df['Text']= df['Text'].str.strip()
# df['Sentiment']= df['Sentiment'].str.strip()
df['User']= df['User'].str.strip()

# remove leading and trailing whitespace characters
df['Platform'] = df['Platform'].str.strip()

df['Country'] = df['Country'].str.strip()


# segment users of each platforms
Facebook=df[df['Platform']=='Facebook']
Twitter=df[df['Platform']=='Twitter']
Instagram=df[df['Platform']=='Instagram']


st.markdown("<hr style='border:2px solid black'>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center;'>Plots</h2>", unsafe_allow_html=True)


# dropdown menu to the sidebar
selected_option = st.sidebar.selectbox('Select a Plot to Display', ['Top Platforms by Total Likes', 'Country', 'Common Words in Text Data'])

# Main content
st.subheader(f'{selected_option}')




#How many likes people n different countries give
no_duplicated = df.drop_duplicates()

no_duplicated.groupby(['Country']).Likes.agg([len, min, max]).sort_values(by='len', ascending=False)

# Plots
def plot_top_platforms_by_likes():
    st.write("## Top Platforms by Total Likes")
    top_likes_platform = df.groupby('Platform')['Likes'].sum().nlargest(10)
    fig, ax = plt.subplots()
    top_likes_platform.plot(kind='bar', ax=ax)
    ax.set_title('Top Platforms by Total Likes')
    ax.set_xlabel('Platform')
    ax.set_ylabel('Total Likes')
    st.pyplot(fig)

def plot_likes_per_country():
    st.write("## Likes per Country")
    likes_per_country = no_duplicated.groupby('Country').Likes.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    likes_per_country.plot(kind='bar', color='#5eAE01', ax=ax)
    ax.set_ylabel('Likes')
    st.pyplot(fig)

def plot_common_words():
    df1 = df.copy()
    st.write("## Common Words in Text Data")
    df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
    top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
    top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])
    
    fig = px.bar(top_words_df,
                 x="count",
                 y="Common_words",
                 title='Common Words in Text Data',
                 orientation='h',
                 width=700,
                 height=700,
                 color='Common_words')
    
    st.plotly_chart(fig)




if selected_option == 'Top Platforms by Total Likes':
    plot_top_platforms_by_likes()
elif selected_option == 'Country':
    plot_likes_per_country()
elif selected_option == 'Common Words in Text Data':
    plot_common_words()




# Draw a line to separate the plots from machine learning part
st.markdown("<hr style='border:2px solid black'>", unsafe_allow_html=True)

# Machine Learning Section Header
st.markdown("<h2 style='text-align:center;'>Machine Learning</h2>", unsafe_allow_html=True)



st.markdown("<p style='text-align:center;'>Please select a machine learning model from the sidebar dropdown and click on the button below it to run the selected model.</p>", unsafe_allow_html=True)


st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

# Machine Learning 

analyzer = SentimentIntensityAnalyzer()

df['Vader_Score'] = df['Clean_Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

df['Sentiment'] = df['Vader_Score'].apply(lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral'))

#split data
X = df['Clean_Text'].values
y = df['Sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#convert text documents into numerical feature vectors(NLP)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)




#dropdown menu for machine learning models
selected_model = st.sidebar.selectbox('Select a Model' ,['Logistic Regression' , 'Random Forest Classifier'])


# Perform the selected machine learning model
if st.sidebar.button('Run Selected Model'):
    if selected_model == 'Logistic Regression':
        logistic_classifier = LogisticRegression(max_iter=50, random_state=42)
        logistic_classifier.fit(X_train_tfidf, y_train)
        # prediction and accuracy
        y_pred_logistic = logistic_classifier.predict(X_test_tfidf)
        accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
        classification_rep_logistic = classification_report(y_test, y_pred_logistic)
        st.write("Logistic Regression Results:")
        st.write(f"Accuracy: {accuracy_logistic}")
        st.write("Classification Report:\n", classification_rep_logistic)
    elif selected_model == 'Random Forest Classifier':
        random_forest_classifier = RandomForestClassifier(random_state=42)
        random_forest_classifier.fit(X_train_tfidf, y_train)        
        y_pred_rf = random_forest_classifier.predict(X_test_tfidf)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        classification_rep_rf = classification_report(y_test, y_pred_rf)
        st.write("\nRandom Forest Results:")
        st.write(f"Accuracy: {accuracy_rf}")
        st.write("Classification Report:\n", classification_rep_rf)




st.markdown("<br>", unsafe_allow_html=True)  
st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True) 


st.markdown("<p style='text-align: center; font-size: 16px;'>Explore the full dataset on Kaggle:</p>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; font-size: 16px;'><a href='https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset' target='_blank'>Kaggle Dataset</a></p>", unsafe_allow_html=True)9
