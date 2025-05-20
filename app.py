import streamlit as st
from rag import load_document, vector_store, graph, MemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="LangGraph PDF Chat", layout="wide")
st.title("📄 Chat with your PDF using LangGraph")

# Upload a file
uploaded_file = st.file_uploader("Upload a PDF, TXT, or CSV file", type=["pdf", "txt", "csv"])

# Session state to store the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False

# Handle document upload
if uploaded_file and not st.session_state.graph_ready:
    with st.spinner("Processing and indexing document..."):
        docs = load_document(uploaded_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        _ = vector_store.add_documents(all_splits)
        st.session_state.graph_ready = True
        st.success("Document indexed and graph ready!")

# Chat interface
if st.session_state.graph_ready:
    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        # Append user message
        st.session_state.messages.append(HumanMessage(content=user_input))

        config = {"configurable": {"thread_id": "streamlit-thread"}}
        with st.spinner("Getting response..."):
            for event in graph.stream(
                {"messages": st.session_state.messages},
                stream_mode="values",
                config=config,
            ):
                output_message = event["messages"][-1]
        
        # Append response
        st.session_state.messages.append(output_message)

        # Display conversation
        st.markdown("### Conversation")
        for msg in st.session_state.messages:
            role = "🧑‍💻 You" if msg.type == "human" else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg.content}")
