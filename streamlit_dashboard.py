import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

from optimized_chatbot import OptimizedRAGChatbot, QueryType
from evaluation_framework import RAGEvaluator, ABTestManager, EvaluationMetric

# Page config
st.set_page_config(
    page_title="RAG Finance Chatbot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
            <style>
                .metric-card {
                    background-color: #f0f2f6;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border-left: 4px solid #1f77b4;
                }
                .improvement-positive {
                    color: #28a745;
                    font-weight: bold;
                }
                .improvement-negative {
                    color: #dc3545;
                    font-weight: bold;
                }
            </style>
            """, unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """
    Initialize chatbot and evaluation componenets.
    """
    try:
        chatbot = OptimizedRAGChatbot()
        evaluator = RAGEvaluator()
        ab_manager = ABTestManager(evaluator)
        return chatbot, evaluator, ab_manager
    
    except Exception as e:
        st.error(f"Failed to initialize components: str{e}")
        return None, None, None
