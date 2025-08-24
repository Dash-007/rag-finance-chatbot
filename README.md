# Stateful RAG-Based Financial Domain Expert Chatbot 🤖💰

A conversational RAG (Retrieval-Augmented Generation) chatbot specialized in financial knowledge, built with LangChain, OpenAI, and Pinecone. The bot provides expert-level financial advice and information by leveraging a curated knowledge base of Investopedia articles and financial terms.

## 🌟 Features

- **Stateful Conversations**: Maintains context across multiple exchanges for natural dialogue flow
- **Domain Expertise**: Specialized in finance and investment topics with curated knowledge base
- **RAG Architecture**: Combines retrieval of relevant documents with generative AI for accurate responses
- **Source Attribution**: Shows which knowledge sources informed each response
- **Interactive CLI**: User-friendly command-line interface with conversation management
- **Robust Error Handling**: Graceful handling of API failures and data loading issues

## 🏗️ Architecture

```
User Query → Vector Search (Pinecone) → Context Retrieval → LLM (GPT-4) → Response
     ↑                                                                        ↓
Chat History ←------- Conversation State Management ←----------------------
```

**Components:**
- **Data Ingestion** (`ingestion.py`): Processes and embeds financial documents into vector database
- **Chatbot Engine** (`stateful-bot.py`): Handles conversations with memory and context retrieval
- **Vector Store**: Pinecone for semantic search of financial knowledge
- **LLM**: OpenAI GPT-4 for natural language generation

## 📊 Knowledge Base

The chatbot's expertise comes from two primary sources:

1. **Investopedia Articles**: Comprehensive financial education content scraped from Investopedia
   - `investopedia_articles_001.csv`
   - `investopedia_articles_002.csv`

2. **Financial Terms Dataset**: Curated from FinRAD (Financial Readability Assessment Dataset)
   - `data_samples.csv` containing financial definitions and terminology

**Total Knowledge Base**: Processed into 1000-character chunks with 200-character overlap for optimal retrieval.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and index

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/stateful-rag-financial-chatbot.git
cd stateful-rag-financial-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
INDEX_NAME=your_pinecone_index_name
```

4. **Prepare your data**
Ensure your CSV files are in the `data/` directory:
```
data/
├── data_samples.csv
├── investopedia_articles_001.csv
└── investopedia_articles_002.csv
```

### Usage

1. **First-time setup - Ingest your data**
```bash
python ingestion.py
```

2. **Start chatting**
```bash
python stateful-bot.py
```

3. **Interactive commands**
- Type your financial questions naturally
- `history` - View conversation history
- `clear` - Reset conversation memory
- `quit` - Exit the chatbot

## 💬 Example Conversation

```
You: What's the difference between stocks and bonds?

🤖 Chatbot: Great question! Stocks and bonds are two fundamental types of investments with key differences:

Stocks represent ownership shares in a company. When you buy stock, you become a partial owner and can benefit from the company's growth through price appreciation and dividends. However, stocks are generally more volatile and risky.

Bonds, on the other hand, are essentially loans you make to companies or governments. They provide more predictable income through regular interest payments and return your principal at maturity. Bonds are typically considered safer but offer lower potential returns.

The choice between them often depends on your risk tolerance and investment timeline. Are you thinking about a specific investment strategy?

💡 Drew from: stocks,bonds

You: I'm 25 and just starting to invest. What should I focus on?
```

## 🔧 Configuration

### Chatbot Parameters

Modify these in `stateful-bot.py`:

```python
bot = Chatbot(
    temperature=0.3,  # Response creativity (0.0-1.0)
    model="gpt-4"     # OpenAI model
)
```

### Retrieval Settings

Adjust search parameters:

```python
retriever=self.vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Number of documents to retrieve
)
```

### Text Splitting

Customize chunking in `ingestion.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200   # Overlap between chunks
)
```

## 📁 Project Structure

```
stateful-rag-financial-chatbot/
├── ingestion.py              # Data processing and vector store creation
├── stateful-bot.py          # Main chatbot application
├── data/                    # Knowledge base CSV files
│   ├── data_samples.csv
│   ├── investopedia_articles_001.csv
│   └── investopedia_articles_002.csv
├── .env                     # Environment variables (not in repo)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Requirements

Create a `requirements.txt` file with:

```txt
langchain
langchain-community
langchain-openai
langchain-pinecone
langchain-text-splitters
openai
pinecone-client
python-dotenv
```

## 🔍 How It Works

### Data Ingestion Process

1. **CSV Loading**: Safely loads financial documents with error handling
2. **Document Processing**: Extracts content and preserves relevant metadata
3. **Text Splitting**: Breaks documents into optimal chunks for retrieval
4. **Embedding Creation**: Generates vector embeddings using OpenAI's text-embedding-3-small
5. **Vector Storage**: Stores embeddings in Pinecone for fast similarity search

### Conversation Flow

1. **Query Processing**: User question is processed and contextualized with chat history
2. **Semantic Search**: Relevant documents are retrieved from the vector database
3. **Context Assembly**: Retrieved documents are combined with conversation history
4. **Response Generation**: LLM generates a natural, contextual response
5. **State Management**: Conversation history is updated for future context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is not licensed.

## 🙏 Acknowledgments

- **Investopedia** for comprehensive financial education content
- **FinRAD Dataset** for financial terminology and definitions
- **LangChain** for the RAG framework
- **OpenAI** for embeddings and language model
- **Pinecone** for vector database capabilities

## 📞 Support

Having issues? Please check the following:

1. Ensure all API keys are correctly set in your `.env` file
2. Verify your Pinecone index exists and is accessible
3. Check that all CSV files are present in the `data/` directory
4. Make sure you've run `ingestion.py` before starting the chatbot

For additional help, please open an issue in the GitHub repository.

---

**Built with ❤️ for financial education and AI-powered assistance**