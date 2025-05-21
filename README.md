# PROJECT: CHATBOT IMPLEMENTATION USING RAG(MESSAGES AND AGENTS)

## Overview:
This project aims at providing the user a back-and-forth conversation with the chatbot that retrieves knowledge from the provided documents(.pdf,.txt,.csv) and then answers the questions based on that document.

This rag uses a sequence of messages instead of having the user input, retrieved context and generated as separate keys in the state. 
## Streamlit Link
    https://buildfastwithai-pfmftvl4jappox9g3pti9zb.streamlit.app/
## Installation
### Steps
1. Clone the repository:
    ```sh
    git clone git@github.com:Krishan098/buildfastwithai.git
    cd buildfastwithai
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
## Usage
Run on Streamlit:
    ```sh
    streamlit run app.py
    ```

## Functions Defined:

### rag.py
1. load_document(file): 

- Use this function to load a document so that it could be further processed. It uses PyPDFLoader, CSVLoader and TextLoader from langchain_community.document_loaders to load the respective document.

- It takes the file name as the argument.

2. retrieve(query):

-  Given a user input, relevant splits are retrieved form storage using a retriever(Vector stores). 

- Vector stores are specialized data stores that enable indexing and retrieving information based on vector representations. These vectors are called embeddings.

-  similarity_search(query,k) takes the search query, creates an embedding, find similar documents and return them as a list of documents. k(int): number of documents to return.

- If relevant content found,the retrieve function returns the metadata and content of each retrieved document otherwise returns "No relevant content found in the PDF."

3. query_or_respond(state: MessagesState):

- This function uses an LLM bound with tools (e.g., a retrieval function) to generate an appropriate response given a sequence of messages from the user and assistant.

- state of application controls what data is input to the application, transferred between steps and output by the application.

- MessagesState is defined with a single messages key which is a list of AnyMessage objects and uses the add_messages reducer.

- returns a dictionary consisting new assistant messages in the format {"messages": [response]}

4. generate(state: MessagesState):

- generates answers

- stores the generated tool messages

- calls the llm to answer according to the given prompt

- returns the System messages and the conversation messages in a dictionary.

- SystemMessage: Message for priming AI behaviour, it is usually passed in as the first of a sequence of input messages.

### app.py

- Creates an interactive interface for the rag model.
- Users can upload a document(.txt,.csv,.pdf).
- After processing users can ask questions based on the document.