import os
import pandas as pd
from pathlib import Path
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv('KEY'))

# Load once globally
e5_model = SentenceTransformer("intfloat/e5-base")

def rephrase_question(question):
    """Rephrase a technical question into a conversational interview style"""
    try:
        prompt = f"""Convert this technical question into a natural, conversational interview style while maintaining its professional tone:
Original: {question}
Make it sound like a senior data scientist asking a candidate during an interview."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error rephrasing question: {str(e)}")
        return question

def get_random_question(category="All"):
    """Get a random question from the dataset and rephrase it"""
    try:
        current_dir = Path(__file__).parent.parent
        data_path = current_dir / 'interview_qa_combined.csv'

        if not data_path.exists():
            return None, None

        df = pd.read_csv(data_path)
        if df.empty:
            return None, None

        if category != "All":
            df = df[df['Category'] == category]

        if df.empty:
            return None, None

        question = df.sample(n=1).iloc[0]
        rephrased_question = rephrase_question(question['Question'])
        return rephrased_question, question['Answer']
    except Exception as e:
        print(f"Error getting random question: {str(e)}")
        return None, None

def evaluate_answer(question, user_answer, ideal_answer):
    """Evaluate user's answer using both sentence embeddings and GPT for comprehensive feedback"""
    try:
        # Define low-effort responses
        low_effort_phrases = {
            "i have no idea", "i don't know", "no idea", "idk", "?", "none",
            "skip", "pass", "don't know", "no answer", "not sure", "not applicable"
        }

        # Normalize and check for low-effort input
        normalized = user_answer.strip().lower()
        if len(normalized) < 5 or normalized in low_effort_phrases or normalized.endswith("?"):
            return f"""Score: 0/10

Ideal Response:
{ideal_answer}"""

        # First, get similarity score using sentence embeddings
        emb_user = e5_model.encode(f"query: {user_answer}", convert_to_tensor=True)
        emb_ideal = e5_model.encode(f"passage: {ideal_answer}", convert_to_tensor=True)
        similarity = util.cos_sim(emb_user, emb_ideal).item()
        similarity_score = int(round(similarity * 10))

        # Create a detailed prompt for GPT that includes the similarity analysis
        prompt = f"""As an expert data science interviewer, evaluate this answer to the following interview question.
The answer has been compared to the ideal response using semantic similarity, which gave a score of {similarity_score}/10.

Question: {question}

User's Answer: {user_answer}

Ideal Answer: {ideal_answer}

Please provide a detailed evaluation that:
1. Considers the semantic similarity score ({similarity_score}/10) but provide your own expert score
2. Identifies key strengths of the answer
3. Points out specific areas for improvement
4. Assesses technical accuracy
5. Evaluates communication clarity
6. Provides specific suggestions for improvement
7. Explains how the answer compares to the ideal response

Format your response as follows:
Score: [your expert score]/10

Semantic Similarity Score: {similarity_score}/10

Strengths:
- [strength 1]
- [strength 2]
...

Areas for Improvement:
- [improvement 1]
- [improvement 2]
...

Technical Accuracy:
[assessment]

Communication Clarity:
[assessment]

Comparison with Ideal Response:
[detailed comparison]

Suggestions:
- [suggestion 1]
- [suggestion 2]
...

Ideal Response:
{ideal_answer}"""

        # Get evaluation from GPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in answer evaluation: {str(e)}")
        return "Error evaluating answer. Please try again."

def get_ai_response(question):
    """Get AI response for user's question using Replit AI"""
    try:
        prompt = f"""As an expert data scientist, provide a detailed answer to this interview question: {question}"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting AI response: {str(e)}")
        return "Error getting response. Please try again."

def get_question_categories():
    """Get list of available question categories"""
    df = load_questions()
    if df is None:
        return ["All"]
    return ["All"] + sorted(df['Category'].unique().tolist())

def load_questions():
    """Load interview questions from CSV file"""
    try:
        current_dir = Path(__file__).parent.parent
        data_path = current_dir / 'interview_qa_combined.csv'

        if not data_path.exists():
            st.error(f"Data file not found at {data_path}")
            return None

        df = pd.read_csv(data_path)
        if df.empty:
            st.error("The dataset is empty")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading questions: {str(e)}")
        return None