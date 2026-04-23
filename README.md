# VSM Based Search Engine using NLP

## Project Overview

This project is a mini topic search engine built for Text Analytics and Natural Language Processing (NLP).

The system takes a user query as input and searches a predefined dataset containing Machine Learning, Deep Learning, Artificial Intelligence, and NLP topics.

The project uses a **Vector Space Model (VSM)** approach with **TF-IDF vectorization** and **Cosine Similarity** to rank the most relevant documents.

---

## Objective

* To understand how Vector Space Model works.
* To convert text data into vectors.
* To compare a user query with stored documents.
* To rank documents based on similarity.
* To deploy the project as a web application.

---

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn

---

## Libraries Used in Code

* `streamlit`
* `pandas`
* `sklearn.feature_extraction.text.TfidfVectorizer`
* `sklearn.metrics.pairwise.cosine_similarity`

---

## Dataset Used

A manually created dataset is used in the code.

It contains topic documents such as:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Regression
* Classification
* Logistic Regression
* Support Vector Machine
* Random Forest
* CNN
* RNN
* LSTM
* Transformer
* Natural Language Processing
* TF IDF
* Bag of Words
* Python
* Pandas
* NumPy
* Scikit Learn

Each topic has a short text description.

---

## Core Concepts Used

### Vector Space Model (VSM)

Documents and user queries are represented as vectors in the same space so they can be compared mathematically.

### TF-IDF

TF-IDF (Term Frequency - Inverse Document Frequency) is used to convert text into numerical vectors and assign importance to useful words.

### Cosine Similarity

Cosine Similarity is used to measure similarity between the user query vector and document vectors.

Higher score means more relevant result.

---

## How the Code Works

1. Topic names and descriptions are stored in a dataset.

2. Title and content are combined into one text field.

3. `TfidfVectorizer()` converts all topic documents into TF-IDF vectors.

4. When a user enters a query, the query is also converted into a vector using the same vectorizer.

5. `cosine_similarity()` compares the query vector with all stored document vectors.

6. Documents are ranked from highest similarity score to lowest.

7. The top result and other relevant documents are shown in the Streamlit interface.

---

## Features Implemented

* Search box for user query
* Search button
* Top ranked result
* Multiple relevant documents
* Similarity score display
* No match message for unrelated queries
* Web deployment using Streamlit

---

## Example Queries

* nlp in python
* support vector machine
* deep learning in cnn
* tf idf and logistic regression
* machine learning

---

## Example Output

Top Result: Natural Language Processing

Relevant Documents:

1. Natural Language Processing
2. Python
3. Tokenization

---

## Deployment

The project is deployed using Streamlit Community Cloud.

Users can open the web link and run the search engine directly in a browser.

---

## Limitations

* Searches only inside the predefined dataset.
* Does not search the internet.
* Dataset size is small.
* Uses keyword/vector similarity, not large language models.

---

## Conclusion

This project demonstrates how a basic search engine can be built using NLP techniques.

By using TF-IDF and Cosine Similarity, the system retrieves relevant topic documents based on the user query and ranks them according to similarity.
