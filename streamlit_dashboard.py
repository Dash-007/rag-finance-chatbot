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
    
def main():
    # Initialize components
    chatbot, evaluator, ab_manager = initialize_components()
    
    if not all([chatbot, evaluator, ab_manager]):
        st.error("Please check your environment variables and try again.")
        return
    
    # Sidebar
    st.sidebar.title("🤖 RAG Finance Chatbot")
    st.sidebar.markdown("---")
    
    #Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["💬 Chat Interface", "📊 Performance Metrics", "🧪 A/B Testing", "📈 Analytics"]
    )
    
    if page == "💬 Chat Interface":
        chat_interface(chatbot)
    elif page == "📊 Performance Metrics":
        chat_interface(evaluator)
    elif page == "🧪 A/B Testing":
        ab_testing_interface(ab_manager, chatbot)
    elif page == "📈 Analytics":
        analytics_dashboard(evaluator)
        
def chat_interface(chatbot):
    """
    Interactive chat interface.
    """
    st.title("💬 Finance Chatbot")
    st.markdown("Ask me anything about finance!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stats" not in st.session_state:
        st.session_state.stats = {"total": 0, "types": {}}
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Response Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Query Type", message["metadata"].get("query_type", "N/A"))
                    with col2:
                        st.metric("Confidence", f"{message['metadata'].get('confidence', 0):.2f}")
                    with col3:
                        st.metric("Sources", message["metadata"].get("sources", 0))
                        
    # Chat input
    if prompt := st.chat_input("Ask a finance question..."):
        # Display user manage
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(chatbot.ask(prompt))
                st.markdown(response["response"])
                
                # Update session stats
                st.session_state.stats["total"] += 1
                query_type = response.get("query_type", "unknown")
                st.session_state.stats["types"][query_type] = st.session_state.stats["types"].get(query_type, 0) + 1
                
                # Store message with metadata
                st.session_state.message.append({
                    "role": "assistant",
                    "content": response["response"],
                    "metadata": {
                        "query_type": query_type,
                        "confidence": response.get("confidence", 0),
                        "sources": response.get("sources", 0)
                    }
                })
                
    # Session statistics
    if st.session_state.stats["total"] > 0:
        st.sidebar.markdown("### Session Stats")
        st.sidebar.metric("Total Queries", st.session_state.stats["total"])
        
        if st.session_state.stats["types"]:
            st.sidebar.markdown("**Query Types:**")
            for qtype, count in st.session_state.stats["types"].items():
                st.sidebar.text(f"{qtype}: {count}")
                
def performance_metrics(evaluator):
    """
    Performance evaluation interface.
    """
    st.title("📊 Performance Metrics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Evaluation Controls")
        
        if st.button("🚀 Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                chatbot = OptimizedRAGChatbot()
                results = asyncio.run(evaluator.evaluate_model(chatbot, "current"))
                filename = evaluator.save_results(results)
                st.success(f"Evaluation complete! Results saved to {filename}")
                st.session_state.latest_results = results
                
    with col2:
        st.markdown("### Test Queries Preview")
        test_queries_df = pd.DataFrame([
            {
                "Query": tq.query,
                "Type": tq.query_type,
                "Difficulty": tq.difficulty
            } for tq in evaluator.test_queries
        ])
        st.dataframe(test_queries_df, use_container_width=True)
        
    # Display results if available
    if "latest_results" in st.session_state:
        st.markdown("---")
        display_evaluation_results(st.session_state.latest_results)
            
def display_evaluation_results(results):
    """
    Display evaluation results with visualizations.
    """
    st.markdown("### 📈 Latest Evaluation Results")
    
    # Aggregate metrics
    avg_metrics = {}
    for metric in EvaluationMetric:
        metric_values = [r.metrics.get(metric.value, 0) for r in results]
        avg_metrics[metric.value] = sum(metric_values) / len(metric_values) if metric_values else 0
        
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Relevance",
            f"{avg_metrics['relevance']:.2f}",
            delta=None
        )
        
    with col2:
        st.metric(
            "Completeness",
            f"{avg_metrics['completeness']:.2f}",
            delta=None
        )
        
    with col3:
        avg_time = sum(r.response_time for r in results) / len(results)
        st.metric(
            "Avg Response Time",
            f"{avg_time:.2f}s",
            delta=None
        )
        
    with col4:
        avg_sources = sum(r.sources_used for r in results) / len(results)
        st.metric(
            "Avg Sources",
            f"{avg_sources:.1f}",
            delta=None
        )
        
    # Detailed results table
    st.markdown(" ### 📋 Detailed Results")
    results_df = pd.DataFrame([
        {
            "Query": r.query[:50] + "..." if len(r.query) > 50 else r.query,
            "Type": r.query_type,
            "Response Time": f"{r.response_time:.2f}s",
            "Relevance": f"{r.metrics.get('relevelace', 0):.2f}",
            "Completeness": f"{r.metrics.get('completeness', 0):.2f}",
            "Sources": r.sources_used
        } for r in results
    ])
    st.dataframe(results_df, use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics by query type
        type_metrics = {}
        for result in results:
            qtype = result.query_type
            if qtype not in type_metrics:
                type_metrics[qtype] = []
            type_metrics[qtype].append(result.metrics.get('relevance', 0))
            
        type_avg = {qtype: sum(metrics)/len(metrics) for qtype, metrics in type_metrics.items()}
        
        fig = px.bar(
            x=list(type_avg.keys()),
            y=list(type_avg.values()),
            title="Average Relevance by Query Type",
            labels={"x": "Query Type", "y": "Relevance Score"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Response time distribution
        times = [r.response_time for r in results]
        fig = px.histogram(
            times,
            title="Response Time Distribution",
            labels={"values": "Response Times (s)", "count": "Frequency"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
