import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data (if not already available)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Required files not found. Make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
    st.stop()

# App title and description
st.title("📩 Email/SMS Spam Classifier")
st.markdown("🚀 Built with Naive Bayes and NLP preprocessing")

# Input field
input_sms = st.text_area("✉️ Enter a message to classify:")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0][result] * 100

        # Display
        if result == 1:
            st.error(f"🚫 This message is **SPAM** with {prob:.2f}% confidence.")
        else:
            st.success(f"✅ This message is **NOT SPAM** with {prob:.2f}% confidence.")

# Footer note
st.markdown("---")
st.markdown(
    "👨‍💻 *This app uses machine learning to classify messages. Use it for educational and testing purposes.*",
    unsafe_allow_html=True
)
