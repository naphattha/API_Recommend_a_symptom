import pandas as pd
import numpy as np
import json
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Load Data ===
df = pd.read_csv("[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment) - ai_symptom_picker.csv")  # if downloaded as CSV

# === 2. Extract base and detail symptoms from summary ===
def extract_symptoms(summary_str):
    try:
        summary = json.loads(summary_str)
        yes_symptoms = summary.get('yes_symptoms', [])
        base = []
        detail = []
        for s in yes_symptoms:
            base.append(s['text'].strip())
            detail.extend([ans.strip() for ans in s.get('answers', [])])
        return base, detail
    except Exception:
        return [], []

df[['base_symptoms', 'detail_symptoms']] = df['summary'].apply(
    lambda x: pd.Series(extract_symptoms(x))
)

# === 3. Add search_term symptoms as context ===
def extract_search_terms(search_term):
    if pd.isna(search_term):
        return []
    return [term.strip() for term in search_term.split(',') if term.strip()]

df['search_symptoms'] = df['search_term'].apply(extract_search_terms)

# === 4. Preprocess other features ===

# One-hot encode gender
ohe_gender = OneHotEncoder()
gender_encoded = ohe_gender.fit_transform(df[['gender']]).toarray()

# Bucketize age
df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 100], labels=False).astype(int)

# Vectorize base symptoms (from summary)
df['base_text'] = df['base_symptoms'].apply(lambda x: ','.join(x))
vectorizer_base = CountVectorizer(tokenizer=lambda txt: txt.split(','), binary=True)
base_matrix = vectorizer_base.fit_transform(df['base_text'])

# Vectorize detail symptoms
df['detail_text'] = df['detail_symptoms'].apply(lambda x: ','.join(x))
vectorizer_detail = CountVectorizer(tokenizer=lambda txt: txt.split(','), binary=True)
detail_matrix = vectorizer_detail.fit_transform(df['detail_text'])

# Vectorize search_term symptoms
df['search_text'] = df['search_symptoms'].apply(lambda x: ','.join(x))
vectorizer_search = CountVectorizer(tokenizer=lambda txt: txt.split(','), binary=True)
search_matrix = vectorizer_search.fit_transform(df['search_text'])

# === 5. Combine all features into single matrix ===
X = sp.hstack([
    gender_encoded,
    df[['age_group']],
    base_matrix,
    detail_matrix,
    search_matrix
])


def predict_from_row(row_dict, top_k=5):
    gender = row_dict['gender']
    age = int(row_dict['age'])
    summary = row_dict['summary']
    search_term = row_dict['search_term']

    # === Extract symptoms from summary ===
    try:
        summary_json = json.loads(summary)
        yes_symptoms = summary_json.get('yes_symptoms', [])
        base_symptoms = [s['text'].strip() for s in yes_symptoms]
        detail_symptoms = []
        for s in yes_symptoms:
            detail_symptoms.extend([ans.strip() for ans in s.get('answers', [])])
    except Exception as e:
        base_symptoms, detail_symptoms = [], []

    # === Extract search terms ===
    if pd.isna(search_term):
        search_terms = []
    else:
        search_terms = [term.strip() for term in search_term.split(',') if term.strip()]

    # === Encode features ===
    gender_vec = ohe_gender.transform(pd.DataFrame({'gender': [gender]})).toarray()
    age_group = pd.cut([age], bins=[0, 20, 30, 40, 50, 100], labels=False)[0]
    age_vec = np.array([[age_group]])

    base_text = ','.join(base_symptoms)
    base_vec = vectorizer_base.transform([base_text])

    detail_text = ','.join(detail_symptoms)
    detail_vec = vectorizer_detail.transform([detail_text])

    search_text = ','.join(search_terms)
    search_vec = vectorizer_search.transform([search_text])

    # === Combine input vector ===
    input_vec = sp.hstack([gender_vec, age_vec, base_vec, detail_vec, search_vec])

    # === Compute similarity ===
    sims = cosine_similarity(input_vec, X).flatten()
    top_indices = sims.argsort()[::-1][1:10]

    # === Aggregate base symptoms from top similar patients ===
    similar_base = []
    for i in top_indices:
        similar_base += df.iloc[i]['base_symptoms']

    # === Filter and recommend ===
    recommended = [
        s for s in pd.Series(similar_base).value_counts().index
        if s != "การรักษาก่อนหน้า"
    ][:top_k]

    return recommended
