import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

from optimized_chatbot import OptimizedRAGChatbot, QueryType
from evaluation_framework import RAGEvaluator, ABTestManager, EvaluationMetric

st.set_page_config(
    page_title="RAG Finance Chatbot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


