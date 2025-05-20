import os
import pypdf as pdf
from typing import List, Dict,Any
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
def load_document(file_Path:str)->List[Dict[str,Any]]:
    """
    This function allows you to load documents from various file formats(.csv,.txt,.pdf).
    
    Args: 
        file_Path: Path of the document.
        
    Returns:
        List of document chunks
    """
    file_extension=os.path.splitext(file_Path)[1].lower()
    
    if file_extension=='.pdf':
        loader=PyPDFLoader(file_Path)
        documents=loader.load()
    if file_extension in ['.txt','.md']:
        loader=TextLoader(file_Path)
        documents=loader.load()
    if file_extension=='.csv':
        loader=CSVLoader(file_Path)
        documents=loader.load()
    else:
        raise ValueError(f'unsupported file format:{file_extension}')
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)
    return text_splitter.split_documents(documents)
    