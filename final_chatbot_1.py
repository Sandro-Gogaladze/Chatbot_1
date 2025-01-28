#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:37:33 2024

@author: sandrogogaladze
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model = "gpt-4o")

# filepath = "/Users/sandrogogaladze/Desktop/what to do/nbg/model risk/docs"
filepath = "/Users/sandrogogaladze/Desktop/PDFS"
loader = PyPDFDirectoryLoader(filepath)

docs = loader.load()

embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents, embeddings)
for d in documents:
    print(d)
    print("\n")
    

template = """"Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


while True:
    question = input('Ask a question: \n')
    if question == "stop":
        break
    else:
        response = retrieval_chain.invoke({"input": question})
        answer = response['answer']
        print(answer)
        