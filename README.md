# 🩺 Symptom Recommender System

This project implements a symptom recommendation system designed to assist in identifying relevant symptoms based on historical patient data. Developed as part of a data science assignment for Agnos, this system aims to provide accurate symptom suggestions, enhancing the efficiency of patient assessment.

## ✨ Features

* **Data-Driven Recommendations:** Recommends symptoms by finding historical patients with similar profiles and complaints.
* **Feature Engineering:** Utilizes gender, age, a summary of symptoms, and user-provided search terms for comprehensive patient profiling.
* **Cosine Similarity:** Employs cosine similarity to measure the likeness between new patient inputs and historical records.
* **FastAPI Interface:** Exposes a robust API endpoint (`/recommend`) for seamless integration with other applications, fulfilling the bonus requirement.

## 🛠️ Technologies Used

* **Core:** Python
* **Data Manipulation:** Pandas, NumPy
* **Numerical Operations:** SciPy (for sparse matrices)
* **Machine Learning:** Scikit-learn (for `OneHotEncoder`, `CountVectorizer`, `cosine_similarity`)
* **API Framework:** FastAPI
* **ASGI Server:** Uvicorn

## 🚀 Getting Started

Follow these instructions to set up and run the symptom recommendation API locally.

### Prerequisites

* Python 3.8+
* The confidential historical patient data file: `[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment) - ai_symptom_picker.csv` (This file is proprietary and not included in the repository).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/naphattha/API_Recommend_a_symptom.git
    cd API_Recommend_a_symptom
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate.bat
    ```

3.  **Install the required packages:**

    ```bash
    pip install pandas numpy scipy scikit-learn fastapi uvicorn
    ```

4.  **Place the Data File:**
    Ensure the confidential historical patient data file named `[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment) - ai_symptom_picker.csv` is placed in the same directory as `symptom_recommendation.py`. **This file is proprietary and should not be shared or committed to the repository.** 

### Running the API

Once you have placed the data file and installed the dependencies, you can run the FastAPI application:

```bash
python api.py
```

You'll see output similar to this:
```bash
INFO:     Uvicorn running on http://0.0.0.0:8000
```
This indicates the API server is running.

### Testing the API
You can test the API by visiting the interactive API documentation or by sending requests using tools like curl or Postman.

Interactive API Documentation:
Open your web browser and go to:
```bash
http://localhost:8000/docs
```
Here, you can see the available endpoints and test them directly.
