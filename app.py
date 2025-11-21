from flask import Flask, render_template,jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os
from operator import itemgetter # REQUIRED: Import itemgetter for RAG chain fix

app= Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embeddings=download_hugging_face_embeddings()

index_name="medical-chatbot"

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings, 
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt + "\n\nUse the following context to answer the user's question. If you do not know the answer, state that you do not have enough information."),
        ("human", "Context: {context}\n\nQuestion: {input}"),
    ]
)

rag_chain = (
    RunnableParallel(
        context=itemgetter("input") | retriever, 
        input=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
    try:
        msg=request.form["msg"]
        print("User query: ",msg)
        response=rag_chain.invoke({"input":msg})
        print("Response: ",response)
        return str(response)
    except Exception as e:
        print(f"Error processing chat message: {e}")
        return jsonify({"error": "Failed to generate response."}), 500


if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)