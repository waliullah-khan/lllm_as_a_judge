import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import anthropic

# Load environment variables
load_dotenv()

# Initialize models
@st.cache_resource
def load_models():
    return {
        'similarity': SentenceTransformer('all-MiniLM-L6-v2'),
        'openai': OpenAI(api_key=os.getenv('OPENAI_API_KEY')),
        'anthropic': anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    }

# LLM Response Generators
def generate_deepseek_response(prompt: str) -> str:
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Deepseek API Error: {str(e)}")
        return ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_perplexity_response(prompt: str) -> str:
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"},
            json={
                "model": "sonar-small-online",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        response_json = response.json()
        
        if 'choices' not in response_json:
            st.error(f"Perplexity API Unexpected Response: {response_json}")
            return ""
            
        return response_json['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        st.error(f"Perplexity API Connection Error: {str(e)}")
    except KeyError as e:
        st.error(f"Perplexity API Response Format Error: {str(e)}")
    except Exception as e:
        st.error(f"Perplexity API Error: {str(e)}")
    return ""

def generate_response(llm: str, prompt: str, clients: dict) -> str:
    try:
        if llm == 'openai':
            response = clients['openai'].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif llm == 'anthropic':
            response = clients['anthropic'].messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif llm == 'deepseek':
            return generate_deepseek_response(prompt)
        
        elif llm == 'perplexity':
            return generate_perplexity_response(prompt)
        
    except Exception as e:
        st.error(f"{llm} Error: {str(e)}")
        return ""

# Evaluation Metrics
def calculate_f1(answer: str, reference: str) -> float:
    """Calculate F1 score between two strings at word level"""
    answer_tokens = set(answer.lower().split())
    reference_tokens = set(reference.lower().split())
    
    if not answer_tokens or not reference_tokens:
        return 0.0
    
    common_tokens = answer_tokens & reference_tokens
    precision = len(common_tokens) / len(answer_tokens)
    recall = len(common_tokens) / len(reference_tokens)
    
    if (precision + recall) == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def calculate_scores(bot_response: str, llm_response: str, model) -> dict:
    bot_embed = model.encode([bot_response])
    llm_embed = model.encode([llm_response])
    
    return {
        'similarity': cosine_similarity(bot_embed, llm_embed)[0][0],
        'f1_score': calculate_f1(bot_response, llm_response)
    }

def main():
    st.title("🤖 LLM as a Judge")
    
    # Initialize models and clients
    models = load_models()
    
    # File upload
    uploaded_file = st.file_uploader("Upload Chat History JSON", type=["json"])
    
    if uploaded_file:
        data = json.load(uploaded_file)
        
        # Process conversations
        conversations = {}
        for msg in data:
            # Process candidate queries
            if msg['sender'] == 'candidate':
                thread_id = str(msg['assistThreadId']['$oid'])
                conversations[thread_id] = {
                    'query': msg['content'],
                    'bot_response': None,
                    'citations': []
                }
            
            # Process bot responses
            elif msg['sender'] == 'bot':
                thread_id = str(msg['assistThreadId']['$oid'])
                if thread_id in conversations:
                    conversations[thread_id]['bot_response'] = msg['content']
                    conversations[thread_id]['citations'] = msg.get('citations', [])

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'query': conv['query'],
                'bot_response': conv['bot_response'],
                'citations': conv['citations']
            }
            for conv in conversations.values()
            if conv['bot_response'] is not None
        ])

        # Add verification step
        if df.empty:
            st.error("No complete conversations found (missing bot responses)")
            return
            
        if 'query' not in df.columns:
            st.error("Query column missing - check JSON structure")
            return

        # Select LLMs to compare
        llm_choices = ['openai', 'anthropic', 'deepseek', 'perplexity']
        selected_llms = st.multiselect("Select LLMs for Comparison", llm_choices)
        
        if selected_llms and st.button("Run Evaluation"):
            with st.spinner("Generating responses and calculating metrics..."):
                # Generate LLM responses
                for llm in selected_llms:
                    df[f'{llm}_response'] = df['query'].apply(
                        lambda x: generate_response(llm, x, models))
                
                # Calculate metrics
                results = []
                for _, row in df.iterrows():
                    row_metrics = {
                        'citation_score': len(row['citations'])/5 if row['citations'] else 0
                    }
                    
                    for llm in selected_llms:
                        metrics = calculate_scores(
                            row['bot_response'],
                            row[f'{llm}_response'],
                            models['similarity']
                        )
                        row_metrics.update({
                            f'{llm}_similarity': metrics['similarity'],
                            f'{llm}_f1': metrics['f1_score']
                        })
                    
                    results.append(row_metrics)
                
                # Combine results
                metrics_df = pd.DataFrame(results)
                final_df = pd.concat([df, metrics_df], axis=1)
                
                # Display results
                st.subheader("Evaluation Metrics")
                st.dataframe(final_df)
                
                # Download button
                st.download_button(
                    label="Download Full Results",
                    data=final_df.to_csv().encode('utf-8'),
                    file_name='llm_evaluation.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()
