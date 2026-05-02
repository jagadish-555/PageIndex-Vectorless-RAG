import streamlit as st
import json
import time
import tempfile
import os
from agents import NavigationAgent

# Ensure these files (pdf_tools.py and indexer.py) are in your project folder
from pdf_tools import extract_pdf_structured
from indexer import build_tree

# --- Page Config ---
st.set_page_config(page_title="PageIndex: Vectorless RAG", layout="wide")
st.title("PageIndex: Vectorless Reasoning Agent")
st.markdown("""
This system uses Agentic Navigation to explore document structures without Vector Databases.
1. **Upload** a document (PDF). 
2. **Index** it to build a hierarchical tree.
3. **Chat** with the agent to extract information and details.
""")

# --- Session State Initialization ---
# We store the agent and tree in session state so they persist between chat turns
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])
    
    if uploaded_file:
        if st.button("Build Reasoning Index"):
            st.session_state.messages = [] # Reset chat for new document
            
            with st.status("Processing Document...", expanded=True) as status:
                try:
                    # 1. Handle File Upload
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # 2. Extract and Index
                    st.write("Parsing document structure...")
                    structured_data = extract_pdf_structured(tmp_path)
                    
                    st.write("Generating AI summaries for document nodes...")
                    tree = build_tree(structured_data)
                    
                    # 3. Initialize the Agent
                    st.session_state.agent = NavigationAgent(tree)
                    
                    os.remove(tmp_path) # Cleanup
                    status.update(label="Indexing Complete", state="complete", expanded=False)
                    st.success("The agent is ready to navigate this document.")
                    
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    status.update(label="Error", state="error")

    if st.session_state.agent:
        st.divider()
        st.caption("Agent Status: Online")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# --- Main Chat Interface ---
if st.session_state.agent is None:
    st.info("Please upload and index a PDF in the sidebar to begin.")
else:
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "trace" in msg:
                with st.expander("View Agent's Reasoning"):
                    for step in msg["trace"]:
                        st.write(f"- {step}")

    # Chat Logic
    if prompt := st.chat_input("Ask about the document (e.g., 'Compare the revenue vs risks')"):
        
        # User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Assistant Message
        with st.chat_message("assistant"):
            with st.status("Agent is navigating...", expanded=True) as status:
                start_time = time.time()
                
                # The agent now performs multi-node retrieval
                answer = st.session_state.agent.navigate(prompt)
                if answer is None:
                    answer = "I could not find relevant information in the document to answer your query."
                
                end_time = time.time()
                
                # Visualizing the trace steps
                for step in st.session_state.agent.trace:
                    st.write(f"- {step}")
                    time.sleep(0.1)
                
                status.update(label=f"Exploration complete in {round(end_time - start_time, 2)}s", state="complete", expanded=False)

            st.markdown(answer)
            
            # Save to history with the full trace
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "trace": list(st.session_state.agent.trace)
            })