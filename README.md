# VSM Based Search Engine using NLP

## Project Overview

This project is a mini search engine developed using **Text Analytics** and **Natural Language Processing (NLP)** concepts.

The system accepts a user query as input and returns the most relevant results related to **Machine Learning** and **NLP** topics.

The project is based on the **Vector Space Model (VSM)**, where both documents and user queries are converted into vectors and compared using similarity measures.

---

## Objective

- To understand the working of Vector Space Model.
- To build a query-based search engine.
- To rank documents based on relevance.
- To deploy the project as a web application.

---

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

---

## Working Principle

1. A dataset of Machine Learning and NLP topics is created.
2. Each document is converted into vector form using **TF-IDF**.
3. User query is also converted into vector form.
4. **Cosine Similarity** is used to compare query with all documents.
5. Top matching results are shown to the user.

---

## Features

- Search engine style interface
- Fast query processing
- Top relevant results
- Handles unrelated queries
- Simple and interactive web app

---

## Example Queries

- what is regression
- explain cnn
- what is nlp
- random forest
- teach me clustering

---

## Project Structure

```text
VSM_Search_Engine_Project/
│── app.py
│── requirements.txt
│── README.md