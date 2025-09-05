# Enterprise RAG Finance Chatbot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-supported-blue)](https://www.docker.com/)

A production-ready conversational RAG (Retrieval-Augmented Generation) system specialized in financial knowledge. Built with LangChain, OpenAI GPT-4, and Pinecone with advanced query classification, A/B testing capabilities, and comprehensive evaluation framework.

## Key Features

### Core Capabilities
- **Advanced Query Classification**: Automatic categorization into definition, calculation, advice, comparison, and general queries
- **Specialized Retrieval Strategies**: Query-type-specific document retrieval for optimal context
- **Stateful Conversations**: Maintains context across multi-turn conversations
- **Source Attribution**: Transparent sourcing of information from knowledge base

### Enterprise Features
- **Configuration Management**: Multiple model configurations for different use cases
- **A/B Testing Framework**: Built-in experimentation platform for model comparison
- **Comprehensive Evaluation**: Automated testing with relevance, accuracy, and performance metrics
- **Interactive Dashboard**: Streamlit-based interface with analytics and testing tools
- **Production Ready**: Error handling, caching, monitoring, and Docker deployment

### Technical Architecture
- **Modular Design**: Separate components for ingestion, retrieval, generation, and evaluation
- **Error Handling**: Robust fallback mechanisms and graceful error recovery
- **Caching System**: In-memory caching for improved response times
- **Monitoring**: Built-in statistics tracking and performance monitoring

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- Pinecone API key and index

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Dash-007/rag-finance-chatbot.git
cd rag-finance-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment setup**
```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_api_key
# PINECONE_API_KEY=your_pinecone_api_key
# INDEX_NAME=your_pinecone_index_name
```

4. **Data ingestion** (first-time setup)
```bash
python ingestion.py
```

5. **Launch dashboard**
```bash
streamlit run streamlit_dashboard.py
```

### Docker Deployment

```bash
# Build image
docker build -t rag-finance-chatbot .

# Run container
docker run -p 8501:8501 --env-file .env rag-finance-chatbot
```

Access the application at `http://localhost:8501`

## Usage Examples

### Basic Chat Interface
```python
import asyncio
from src.optimized_chatbot import create_chatbot

# Initialize chatbot
bot = create_chatbot("baseline")

# Ask questions
async def demo():
    result = await bot.ask("What is compound interest?")
    print(result['response'])
    
asyncio.run(demo())
```

### Configuration Management
```python
from src.config_manager import ConfigManager, ModelConfig

# Create custom configuration
config = ModelConfig(
    name="custom_config",
    model="gpt-4",
    temperature=0.3,
    retrieval_k=6
)

# Apply configuration
config_manager = ConfigManager()
config_manager.add_model_config(config)
bot = create_chatbot("custom_config")
```

### A/B Testing
```python
import asyncio
from src.evaluation_framework import ABTestManager, RAGEvaluator

async def run_test():
    evaluator = RAGEvaluator()
    ab_manager = ABTestManager(evaluator)
    
    # Compare configurations
    results = await ab_manager.run_ab_test(
        model_a=create_chatbot("baseline"),
        model_b=create_chatbot("creative"),
        test_name="temperature_comparison"
    )
    print(f"Winner: {results['winner']}")
    
asyncio.run(run_test())
```

## Knowledge Base

The chatbot leverages a curated financial knowledge base including:

- **Investopedia Articles**: Comprehensive financial education content (2 CSV files)
- **Financial Terms Dataset**: Specialized terminology and definitions
- **Processing Pipeline**: Automated chunking and embedding generation

**Total Knowledge Base**: 1000+ financial documents processed into optimized chunks for semantic search.

## Configuration Options

### Model Configurations
- **Baseline**: Balanced performance (temperature: 0.7, k: 4)
- **Creative**: Higher creativity (temperature: 0.9, k: 4)
- **Conservative**: Focused responses (temperature: 0.3, k: 6)
- **Fast**: Quick responses using GPT-3.5-turbo (k: 3)

### Retrieval Settings
- **Embedding Model**: text-embedding-3-small (512 dimensions)
- **Chunk Size**: 1000 characters with 200-character overlap
- **Retrieval Count**: Configurable (default: 4 documents)

## Evaluation Metrics

The system includes comprehensive evaluation across multiple dimensions:

- **Relevance**: Keyword coverage and semantic alignment
- **Completeness**: Response thoroughness and depth
- **Response Time**: Performance optimization tracking
- **Accuracy**: Domain-specific correctness validation

## API Reference

### Core Methods

#### OptimizedRAGChatbot
```python
async def ask(question: str) -> Dict[str, Any]
```
Process a user question and return comprehensive response with metadata.

**Returns:**
- `response`: Generated answer
- `query_type`: Classified query category
- `confidence`: Classification confidence score
- `sources`: Number of documents used
- `response_time`: Processing time in seconds

#### QueryClassifier
```python
def classify(query: str) -> QueryContext
```
Analyze and categorize user queries for optimal retrieval strategy.

#### AdvancedRetriever
```python
async def retrieve(query: str, context: QueryContext) -> List[Document]
```
Perform specialized document retrieval based on query classification.

## Testing

Run the comprehensive test suite:

```bash
# All tests
python -m pytest test_suite.py -v

# Specific test categories
python -m pytest test_suite.py::TestQueryClassifier -v
python -m pytest test_suite.py::TestConfigManager -v
python -m pytest test_suite.py::TestOptimizedChatbot -v
```

## Performance Benchmarks

Based on evaluation framework testing with default configurations:

| Configuration | Avg Relevance | Avg Response Time | Completeness |
|---------------|---------------|-------------------|--------------|
| Baseline      | 0.82          | 2.1s             | 0.78         |
| Creative      | 0.79          | 2.3s             | 0.81         |
| Conservative  | 0.85          | 1.9s             | 0.76         |
| Fast          | 0.77          | 1.4s             | 0.73         |

## Development

### Setup Development Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run formatting
python -m black src/ test_suite.py streamlit_dashboard.py

# Run linting
python -m flake8 src/ test_suite.py --max-line-length=88
```

### Adding New Features
1. Implement in appropriate module (`src/`)
2. Add tests in `test_suite.py`
3. Update configuration if needed
4. Update documentation

## Deployment

### Local Development
```bash
streamlit run streamlit_dashboard.py
```

### Docker Production
```bash
docker build -t rag-finance-chatbot .
docker run -d -p 8501:8501 --env-file .env --name finance-bot rag-finance-chatbot
```

### Environment Variables
Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `INDEX_NAME`: Your Pinecone index name

## Troubleshooting

### Common Issues

**API Key Errors**
- Verify environment variables are set correctly
- Check API key validity and permissions

**Vector Store Connection**
- Confirm Pinecone API key and index name
- Ensure index exists and is accessible

**Performance Issues**
- Monitor cache usage in dashboard
- Adjust `retrieval_k` parameter for speed vs accuracy trade-off

**Memory Issues**
- Reduce chunk size in `ingestion.py`
- Clear conversation history periodically

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Contributing

1. Fork the repository
2. Create feature branch (git checkout -b feature/new-feature)
3. Run tests (python -m pytest test_suite.py)
4. Commit changes (git commit -m 'Add new feature')
5. Push to branch (git push origin feature/new-feature)
6. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LangChain** for the RAG framework foundation
- **OpenAI** for language models and embeddings
- **Pinecone** for vector database infrastructure
- **Streamlit** for the interactive dashboard framework

---

**Built for enterprise financial advisory applications with production-ready AI engineering practices.** 