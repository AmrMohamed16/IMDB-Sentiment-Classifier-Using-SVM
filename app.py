import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the vectorizer and trained model
try:
    vector_form = pickle.load(open('vector.pkl', 'rb'))
    load_model = pickle.load(open('trained_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Required files ('vector.pkl' or 'trained_model.pkl') not found. Please generate them first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading pickled files: {e}")
    st.stop()

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

# Function for text preprocessing
def stemming(content):
    content = re.sub('[^a-zA-Z ]', '', content)  # Remove non-alphabetic characters
    content = content.lower()  # Convert to lowercase
    words = content.split()
    stemmed_words = [port_stem.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(stemmed_words)

# Function for sentiment prediction
def sentiment(review):
    review = stemming(review)  # Preprocess the input text
    input_data = [review]
    try:
        vector_form1 = vector_form.transform(input_data)  # Transform input data to TF-IDF vector
        prediction = load_model.predict(vector_form1)  # Predict sentiment
        return prediction[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Streamlit app interface
if __name__ == '__main__':
    st.title('Sentiment Classification App')
    st.subheader("Input the review content below")
    sentence = st.text_area("Enter your review here", "", height=200)

    predict_btt = st.button("Predict")
    if predict_btt:
        if sentence.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            prediction_class = sentiment(sentence)
            if prediction_class == 0:
                st.warning('Negative Sentiment')
            elif prediction_class == 1:
                st.success('Positive Sentiment')
            else:
                st.error('Unable to determine sentiment.')
