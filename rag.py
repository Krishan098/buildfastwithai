import os
os.environ.get("COHERE_API_KEY")
from langchain.chat_models import init_chat_model

llm = init_chat_model("command-r-plus", model_provider='cohere')

from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},    
)

from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file) -> list[Document]:
    '''
        
    Loads the uploaded document for further processing.
        
    Args:
        file: the file name.
    
    Returns:
        a list of document objects(class for storing a peice of text and associated metadata), 
        where document represents chunk of text.

    '''
    file_path = f'temp/{file.name}'
    os.makedirs("temp", exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(file.read())
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, TXT, or CSV.")

    return loader.load()

from langgraph.graph import MessagesState, StateGraph
graph_builder = StateGraph(MessagesState)

from langchain_core.tools import tool

@tool
def retrieve(query: str) -> str:
    """
    
    Retrieve relevant information from the PDF.
    
    Args:
        query: Text to look up documents similar to.
    
    Returns:
        if relevant content found, returns the metadata and content of each retrieved document.
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)#returns a list of documents after creating an embedding and finding similar documents for the query. 
    if not retrieved_docs:
        return "No relevant content found in the PDF."
    
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

#Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """
    
    Generate tool call for retrieval or respond.
    
    Args:
        state (MessagesState): The current message state, typically a list or structure 
                               containing the conversation history (user and assistant messages).
    
    Returns:
        A dictionary containing new assistant messages.                           
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

#Execute the retrieval
tools = ToolNode([retrieve])

#Generate a response 
def generate(state: MessagesState):
    """
    
    Generate answer.
    
    Args:
        state (MessagesState): The current message state, typically a list or structure 
                               containing the conversation history (user and assistant messages).
    
    Returns:
        A dictionary containing system messages and converation messages.
    """
    #Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    #format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import tools_condition

#3 nodes used:
#1. fields the user input, either generating a query for the retriever or responding directly
#2. for the retrieval tool
#3. for generating th final response using the retrieved context
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    '''Used ChatGPT for this part😃'''
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    from IPython.display import Image, display

    # Step 1: Load and split PDF
    docs = load_document(open("t.pdf", "rb"))  # ensure binary mode
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(all_splits)

    # Step 2: Create ReAct agent
    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
    display(Image(agent_executor.get_graph().draw_mermaid_png()))

    # Step 3: Define list of questions
    questions = [
        "What is the pdf about?",
        "List the key steps mentioned in the document.",
        "Who is the intended audience for this document?",
        "Are there any important deadlines or dates?",
        "What actions are expected from the reader?",
    ]

    config = {"configurable": {"thread_id": "qa-thread-001"}}
    qa_pairs = []

    # Step 4: Ask questions and collect answers
    for question in questions:
        final_answer = ""

        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config=config,
        ):
            msg = event["messages"][-1]
            msg.pretty_print()
            if msg.type == "ai":
                final_answer = msg.content

        qa_pairs.append((question, final_answer))

    # Step 5: Save Q&A pairs to output.txt
    with open("output.txt", "w", encoding="utf-8") as f:
        for i, (q, a) in enumerate(qa_pairs, 1):
            f.write(f"Q{i}: {q}\nA{i}: {a}\n\n")

