# -*- coding: utf-8 -*-
"""Installation and Import"""

!pip install langchain openai chromadb tiktoken pypdf panel gradio --upgrade Pillow

import os 
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
import gradio as gr

"""Storage"""

os.environ["OPENAI_API_KEY"]="YOUR_OPENAI_API_KEY"

"""QnA Bot"""

def qa(query):
    # load document
    loader = PyPDFLoader('/content/sui.pdf')
    documents = loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # create the vector store to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity")

    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="map_reduce", retriever=retriever)
    result = qa({"query": query})
    print(result['result'])
    return result

"""QnA Bot Interface"""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []

    def user(user_message, history):
        print("User message:", user_message)
        print("Chat history:", history)

        # Get response from QA function
        response = qa(user_message)

        # Append user message and response to chat history
        history.append((user_message, response["result"]))
        print("Updated chat history:", history)
        return "", history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)