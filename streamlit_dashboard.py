import os
import sys
import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

# For docker compatibility
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.optimized_chatbot import OptimizedRAGChatbot, QueryType
from src.config_manager import ConfigManager, ModelConfig
from src.evaluation_framework import RAGEvaluator, ABTestManager, EvaluationMetric

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
        st.error(f"Failed to initialize components: {str(e)}")
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
        performance_metrics(evaluator)
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
                st.session_state.messages.append({
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
            "Relevance": f"{r.metrics.get('relevance', 0):.2f}",
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
        
def ab_testing_interface(ab_manager, base_chatbot):
    """
    A/B testing interface.
    """
    st.title("🧪 A/B Testing")
    st.markdown("Compare different model configurations")
    
    # Test configuration
    with st.expander("🔧 Configure A/B Test", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            test_name = st.text_input("Test Name", value="temperature_comparison")
            
        with col2:
            model_b_name = st.text_input("Model B Name", value="variant")
            
        # Model B configuration
        st.markdown("**Model B Condiguration (Variant):**")
        col3, col4 = st.columns(2)
        with col3:
            model_a_name = st.text_input("Model A Name", value="baseline")
        with col4:
            model_b = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"], index=0)
            temp_b = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            
    # Run test
    if st.button("🚀 Run A/B Test", type="primary"):
        with st.spinner("Running A/B test... This may take a few minutes."):
            try:
                # Create temporary variant chatbot
                config_manager = ConfigManager()
                
                variant_config = ModelConfig(
                    name="streamlit_variant",
                    model=model_b,
                    temperature=temp_b,
                    embedding_model="text-embedding-3-small",
                    embedding_dimensions=512,
                    retrieval_k=4
                )
                
                config_manager.add_model_config(variant_config)
                
                variant_chatbot = OptimizedRAGChatbot("streamlit_variant")
                
                # Run comparison
                results = asyncio.run(ab_manager.run_ab_test(
                    base_chatbot,
                    variant_chatbot,
                    test_name,
                    model_a_name,
                    model_b_name
                ))
                
                st.session_state.ab_results = results
                st.success("A/B test completed!")
                
            except Exception as e:
                st.error(f"Error running A/B test: {str(e)}")
                
    # Display results
    if "ab_results" in st.session_state:
        st.markdown("---")
        display_ab_results(st.session_state.ab_results)
        
def display_ab_results(results):
    """
    Display A/B test results.
    """
    st.markdown("### 📊 A/B Test Results")
    
    # Winner announcement
    winner = results["winner"]
    if winner != "tie":
        st.success(f"🏆 Winner: **{winner}**")
    else:
        st.info("🤝 Result: **Tie**")
        
    # Comparison metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {results['model_a']['name']} (Baseline)")
        metrics_a = results['model_a']['metrics']
        for metric, value in metrics_a.items():
            st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
        st.metric("Avg Response Time", f"{results['model_a']['avg_response_time']:.2f}s")
        
    with col2:
        st.markdown(f"### {results['model_b']['name']} (Variant)")
        metrics_b = results['model_b']['metrics']
        for metric, value in metrics_b.items():
            if metric in results['improvements']:
                delta = results['improvements'][metric]['absolute_diff']
                st.metric(
                    metric.replace('-', ' ').title(),
                    f"{value:.3f}",
                    delta=f"{delta:+.3f}"
                )
            else:
                st.metric(metric.replace('-', ' ').title(), f"{value:.3f}")
                
        time_diff = results['model_b']['avg_response_time'] - results['model_a']['avg_response_time']
        st.metric(
            "Avg Response Time",
            f"{results['model_b']['avg_response_time']:.2f}s",
            delta=f"{time_diff:+.2f}s"
        )
        
        # Improvements summary
        st.markdown("### 📈 Performance Changes")
        improvements_df = pd.DataFrame([
            {
                "Metric": metric.replace('-', ' ').title(),
                "Absolute Change": f"{data['absolute_diff']:+.3f}",
                "Percent Change": f"{data['percent_change']:+.1f}%"
            } for metric, data in results['improvements'].items()
        ])
        
        st.dataframe(improvements_df, use_container_width=True)
        
def analytics_dashboard(evaluator):
    """
    Analytics and insights dashboard.
    """
    st.title("📈 Analytics Dashboard")
    
    # Load historical results
    results_dir = Path(evaluator.results_dir)
    result_files = list(results_dir.glob("*.json"))
    
    if not result_files:
        st.info("No evaluation results found. Run some evaluations first!")
        
    # File selector
    selected_file = st.selectbox(
        "Select Results File:",
        [f.name for f in result_files],
        index=0
    )
    
    if selected_file:
        results = evaluator.load_results(selected_file)
        
        # Overview metrics
        st.markdown("### 📊 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(results))
            
        with col2:
            avg_time = sum(r.response_time for r in results) / len(results)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
        with col3:
            query_types = len(set(r.query_type for r in results))
            st.metric("Query Types", query_types)
            
        with col4:
            avg_relevance = sum(r.metrics.get('relevance', 0) for r in results)
            st.metric("Avg Relevance", f"{avg_relevance:.2f}")
            
        # Time series analysis
        st.markdown("### 📈 Performance Over Time")
        
        # Create time series data
        df = pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "response_time": r.response_time,
                "relevance": r.metrics.get('relevance', 0),
                "query_type": r.query_type
            } for r in results
        ])
        
        # Performance trends
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df,
                x="timestamp",
                y="response_time",
                color="query_type",
                title="Response Time Trends"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.scatter(
                df,
                x="timestamp",
                y="relevance",
                color="query_type",
                title="Relevance Score Trends"
            )
            st.plotly_chart(fig, use_container_width=True)
            
if __name__ == "__main__":
    main()