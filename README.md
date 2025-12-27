# Autonomous AI Research Agent

An intelligent AI agent that autonomously searches the web, explores Wikipedia, and retrieves research papers from arXiv to answer your questions with cited sources.

🔗 **[Live Demo](https://search-engine-ai-tanvir.streamlit.app/)**

## Features

- 🌐 **Web Search**: Real-time search using DuckDuckGo
- 📚 **Wikipedia Integration**: Access to millions of Wikipedia articles
- 🔬 **arXiv Research**: Retrieve academic papers and research
- 🤖 **Autonomous Agent**: Intelligently selects and uses appropriate tools
- 💬 **Interactive Chat**: Natural conversation interface
- 🔄 **Streaming Responses**: Real-time agent thought process visualization

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.1-8B)
- **Agent Framework**: LangChain
- **Search Tools**: DuckDuckGo, Wikipedia API, arXiv API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autonomous-ai-agent.git
cd autonomous-ai-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
# Create a .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter your Groq API key in the sidebar (or set it in `.env`)
2. Type your question in the chat input
3. Watch as the agent autonomously:
   - Decides which tools to use
   - Searches relevant sources
   - Synthesizes information
   - Provides a comprehensive answer

## Example Queries

- "What is machine learning?"
- "Latest research on quantum computing"
- "Explain the theory of relativity"
- "Current developments in renewable energy"

## Tools Available

- **DuckDuckGo Search**: General web search for current information
- **Wikipedia**: Encyclopedia articles and summaries
- **arXiv**: Academic papers and research publications

## Requirements
```
streamlit
langchain
langchain-groq
langchain-community
duckduckgo-search
wikipedia
arxiv
python-dotenv
```

## Configuration

- **Model**: llama-3.1-8b-instant
- **Streaming**: Enabled for real-time responses
- **Wikipedia Results**: Top 1 result, 200 characters
- **arXiv Results**: Top 1 result, 200 characters

## License

MIT License

## Author

**Tanvir Ahammed**  
📧 Email: tanvir7535@gmail.com

Built with ❤️ using Streamlit, LangChain, and Groq
