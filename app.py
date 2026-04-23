import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Dataset
# ----------------------------
data = [
["Regression","Regression predicts continuous values such as salary house price marks."],
["Classification","Classification predicts categories labels or classes."],
["Clustering","Clustering groups similar data points using unsupervised learning."],
["Random Forest","Random Forest combines many decision trees to improve accuracy."],
["Decision Tree","Decision Tree uses conditions and branches for prediction."],
["Support Vector Machine","SVM is used for classification and regression tasks."],
["Neural Network","Neural Network is inspired by human brain using layers and neurons."],
["Convolutional Neural Network","CNN is used for image recognition and computer vision."],
["Recurrent Neural Network","RNN is used for sequence data such as text and speech."],
["LSTM","LSTM is advanced RNN for long term dependencies."],
["Transformer","Transformer uses attention mechanism for language models."],
["Natural Language Processing","NLP helps computers understand and generate human language."],
["Tokenization","Tokenization splits text into tokens or words."],
["TF IDF","TF IDF measures importance of words in documents."],
["Python","Python is popular for machine learning and data science."]
]

df = pd.DataFrame(data, columns=["title","content"])
df["text"] = df["title"] + " " + df["content"]

# ----------------------------
# VSM Model
# ----------------------------
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
document_vectors = vectorizer.fit_transform(df["text"])

# ----------------------------
# Search Function
# ----------------------------
def search_engine(query):

    query = query.strip().lower()

    if query == "":
        return "Please enter a search query."

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_vectors).flatten()

    priority_map = {
        "nlp":"Natural Language Processing",
        "cnn":"Convolutional Neural Network",
        "rnn":"Recurrent Neural Network",
        "lstm":"LSTM",
        "svm":"Support Vector Machine"
    }

    for key, title_name in priority_map.items():
        if key in query:
            idx = df[df["title"] == title_name].index[0]
            scores[idx] += 0.60

    ranked = scores.argsort()[::-1]
    top_results = [i for i in ranked if scores[i] > 0.08][:3]

    if len(top_results) == 0:
        return "No relevant ML/NLP document matched your query."

    best = top_results[0]

    output = f"Top Result: {df.iloc[best]['title']}\n\n"
    output += f"Summary: {df.iloc[best]['content']}\n\n"
    output += "Top Matches:\n"

    for rank, i in enumerate(top_results, start=1):
        output += f"{rank}. {df.iloc[i]['title']} ({round(scores[i]*100,2)}%)\n"

    return output

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("VSM Based Search Engine")
st.write("Search Machine Learning and NLP topics using Vector Space Model.")

query = st.text_input("Enter your query:")

if st.button("Search"):
    result = search_engine(query)
    st.text(result)