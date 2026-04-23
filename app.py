# ==========================================================
# FINAL app.py
# VSM SEARCH ENGINE (Improved UI + Better Ranking + Multiple Results)
# Replace old app.py بالكامل with this code
# ==========================================================

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------------
st.set_page_config(
    page_title="VSM Search Engine",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------------------------------------
# CUSTOM CSS (Better UI)
# ----------------------------------------------------------
st.markdown("""
<style>
.main-title{
    font-size:52px;
    font-weight:800;
    color:#1f2937;
    margin-bottom:5px;
}
.sub-title{
    font-size:20px;
    color:#4b5563;
    margin-bottom:25px;
}
.result-box{
    background:#f8fafc;
    padding:18px;
    border-radius:12px;
    border:1px solid #dbeafe;
    margin-bottom:12px;
}
.top-box{
    background:#eff6ff;
    padding:22px;
    border-radius:14px;
    border:2px solid #60a5fa;
    margin-bottom:18px;
}
.rank-title{
    font-size:22px;
    font-weight:700;
}
.small-text{
    color:#475569;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# DATASET (Documents)
# ----------------------------------------------------------
data = [
["Regression","Regression predicts continuous values such as salary house price marks."],
["Linear Regression","Linear Regression uses a straight line relationship between variables."],
["Logistic Regression","Logistic Regression is used for binary classification problems."],
["Classification","Classification predicts labels or categories."],
["Decision Tree","Decision Tree uses branches and rules for prediction."],
["Random Forest","Random Forest combines multiple trees for better accuracy."],
["Support Vector Machine","SVM is used for classification and regression tasks."],
["K Nearest Neighbors","KNN predicts using nearest similar data points."],
["Clustering","Clustering groups similar data points using unsupervised learning."],
["K Means","K Means divides data into k clusters."],
["Neural Network","Neural Network uses neurons and layers inspired by brain."],
["Deep Learning","Deep Learning uses multiple neural network layers."],
["Convolutional Neural Network","CNN is used for image recognition and computer vision."],
["Recurrent Neural Network","RNN is used for text speech and sequence data."],
["LSTM","LSTM handles long term dependencies in sequential data."],
["Transformer","Transformer uses attention mechanism in language models."],
["Natural Language Processing","NLP helps computers understand and generate human language."],
["Tokenization","Tokenization splits text into tokens or words."],
["Stopword Removal","Removes common words such as is the and of."],
["Stemming","Stemming reduces words to root form."],
["Lemmatization","Lemmatization converts words into meaningful base form."],
["TF IDF","TF IDF measures importance of words in documents."],
["Bag of Words","Bag of Words converts text into frequency vectors."],
["Python","Python is popular for machine learning and NLP projects."],
["Pandas","Pandas is used for dataframe and data analysis."],
["NumPy","NumPy is used for arrays and numerical computing."],
["Scikit Learn","Scikit Learn provides machine learning algorithms."]
]

df = pd.DataFrame(data, columns=["title","content"])
df["text"] = df["title"] + " " + df["content"]

# ----------------------------------------------------------
# VSM MODEL
# ----------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1,2)
)

document_vectors = vectorizer.fit_transform(df["text"])

# ----------------------------------------------------------
# SEARCH FUNCTION
# ----------------------------------------------------------
def search_engine(query):

    query = query.strip().lower()

    if query == "":
        return None, []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_vectors).flatten()

    # Smart boosts for abbreviations
    boost_map = {
        "nlp":"Natural Language Processing",
        "cnn":"Convolutional Neural Network",
        "rnn":"Recurrent Neural Network",
        "svm":"Support Vector Machine",
        "knn":"K Nearest Neighbors",
        "lstm":"LSTM",
        "tfidf":"TF IDF",
        "tf idf":"TF IDF"
    }

    for key, value in boost_map.items():
        if key in query:
            idx = df[df["title"] == value].index[0]
            scores[idx] += 0.60

    ranked = scores.argsort()[::-1]

    # threshold
    top_results = [i for i in ranked if scores[i] > 0.22][:5]

    if len(top_results) == 0:
        return "No relevant ML/NLP document matched your query.", []

    best = top_results[0]

    top_doc = {
        "title": df.iloc[best]["title"],
        "content": df.iloc[best]["content"],
        "score": round(scores[best]*100,2)
    }

    others = []

    for i in top_results:
        others.append({
            "title": df.iloc[i]["title"],
            "content": df.iloc[i]["content"],
            "score": round(scores[i]*100,2)
        })

    return top_doc, others

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.markdown('<div class="main-title">🔍 VSM Based Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Search Machine Learning and NLP topics using Vector Space Model</div>', unsafe_allow_html=True)

query = st.text_input("Enter your query:", placeholder="Example: nlp in python / support vector machine / regression")

if st.button("Search", use_container_width=True):

    top_result, results = search_engine(query)

    # No match
    if isinstance(top_result, str):
        st.error(top_result)

    else:
        # Top Result
        st.markdown(f"""
        <div class="top-box">
        <div class="rank-title">🏆 Top Result: {top_result['title']}</div>
        <br>
        <b>Summary:</b> {top_result['content']}<br><br>
        <b>Similarity Score:</b> {top_result['score']}%
        </div>
        """, unsafe_allow_html=True)

        # Other Results
        st.subheader("📄 Relevant Documents")

        rank = 1
        for item in results:
            st.markdown(f"""
            <div class="result-box">
            <b>{rank}. {item['title']}</b> ({item['score']}%)<br>
            <span class="small-text">{item['content']}</span>
            </div>
            """, unsafe_allow_html=True)
            rank += 1
