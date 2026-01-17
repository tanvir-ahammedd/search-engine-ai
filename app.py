import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide"
)

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "Hi, I'm a chatbot that can search the web, explore Wikipedia, and retrieve research papers from arXiv. How can I help you?",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": []
        }
    ]

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

# Sidebar
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Stats")
st.sidebar.metric("Total Queries", st.session_state["query_count"])
st.sidebar.metric("Messages", len(st.session_state["messages"]))

# Export functionality
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Export Chat")

def export_chat_history():
    """Export chat history as formatted text"""
    export_text = "AI Research Assistant - Chat History\n"
    export_text += "=" * 50 + "\n\n"
    
    for msg in st.session_state["messages"]:
        role = msg["role"].upper()
        content = msg["content"]
        timestamp = msg.get("timestamp", "N/A")
        sources = msg.get("sources", [])
        
        export_text += f"[{timestamp}] {role}:\n{content}\n"
        
        if sources:
            export_text += f"\nSources used:\n"
            for source in sources:
                export_text += f"  • {source}\n"
        
        export_text += "\n" + "-" * 50 + "\n\n"
    
    return export_text

if st.sidebar.button("📥 Download Chat History"):
    if len(st.session_state["messages"]) > 1:
        chat_export = export_chat_history()
        st.sidebar.download_button(
            label="💾 Download TXT",
            data=chat_export,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.sidebar.warning("No chat history to export yet!")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "Chat history cleared! How can I help you?",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": []
        }
    ]
    st.session_state["query_count"] = 0
    st.rerun()

# Main content
st.title("🔎 AI Research Assistant")
st.markdown("""
An intelligent agent that searches the web, explores Wikipedia, and retrieves research papers from arXiv to answer your questions with cited sources.
""")

# Query suggestions
st.markdown("### 💡 Try these example queries:")
col1, col2, col3 = st.columns(3)

example_queries = [
    "What is machine learning?",
    "Latest developments in quantum computing",
    "Explain transformer architecture in AI",
    "What is climate change?",
    "Recent research on CRISPR technology",
    "History of artificial intelligence"
]

for idx, col in enumerate([col1, col2, col3]):
    with col:
        if st.button(example_queries[idx], key=f"example_{idx}", use_container_width=True):
            st.session_state["selected_query"] = example_queries[idx]
            st.rerun()
        if st.button(example_queries[idx + 3], key=f"example_{idx + 3}", use_container_width=True):
            st.session_state["selected_query"] = example_queries[idx + 3]
            st.rerun()

st.markdown("---")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg['content'])
        
        # Show timestamp and sources
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources Used"):
                for source in msg["sources"]:
                    st.markdown(f"• {source}")
        
        if msg.get("timestamp"):
            st.caption(f"🕐 {msg['timestamp']}")

# Handle selected query from examples
if "selected_query" in st.session_state:
    prompt = st.session_state["selected_query"]
    del st.session_state["selected_query"]
else:
    prompt = st.chat_input(placeholder="Ask me anything...")

# Process user input
if prompt:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar to continue.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": []
    })
    st.session_state["query_count"] += 1
    
    with st.chat_message("user"):
        st.write(prompt)
        st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate response
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = create_agent(
        llm,
        tools,
        system_prompt="""
            You are a helpful research assistant.
            Use the available tools to search for accurate information.
            
            After gathering information:
            1. Provide a clear, comprehensive answer
            2. Be specific and cite key facts
            3. If you used tools, briefly mention which sources you consulted
            
            Keep your response natural and conversational.
            """
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Track sources used
        sources_used = []
        
        # Custom callback to track tool usage
        class SourceTracker:
            def __init__(self):
                self.sources = []
        
        source_tracker = SourceTracker()
        
        try:
            with st.spinner("🔍 Searching for information..."):
                messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
                
                result = search_agent.invoke(
                    {"messages": messages},
                    config={"callbacks": [st_cb]}
                )
                
                response = result["messages"][-1].content
                
                # Extract sources from the agent's intermediate steps
                if "messages" in result:
                    for msg in result["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get('name', 'Unknown')
                                if tool_name == "Search":
                                    sources_used.append("🌐 Web Search (DuckDuckGo)")
                                elif tool_name == "arxiv":
                                    sources_used.append("📄 arXiv Research Papers")
                                elif tool_name == "wikipedia":
                                    sources_used.append("📖 Wikipedia")
                
                # Remove duplicates
                sources_used = list(set(sources_used))
                
                # If no sources detected, add a default
                if not sources_used:
                    sources_used = ["💭 Knowledge Base"]
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.session_state.messages.append({
                    'role': 'assistant', 
                    "content": response,
                    "timestamp": timestamp,
                    "sources": sources_used
                })
                
                st.write(response)
                
                # Display sources
                if sources_used:
                    with st.expander("📚 Sources Used"):
                        for source in sources_used:
                            st.markdown(f"• {source}")
                
                st.caption(f"🕐 {timestamp}")
                
        except Exception as e:
            error_msg = f"❌ An error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                'role': 'assistant', 
                "content": error_msg,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sources": []
            })

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Built with LangChain, Groq, and Streamlit | AI Research Assistant</p>
</div>
""", unsafe_allow_html=True)
