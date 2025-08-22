from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from google import genai
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import time
import uuid
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore


load_dotenv()


# Get API key and index name
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")   # sometimes optional in new Pinecone client
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key= PINECONE_API_KEY)
index_name = PINECONE_INDEX_NAME
index = pc.Index(index_name)



#Models & VectorStore & Prompt Initialization
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

vector_store = None 
namespace = None

description_generating_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    transport="rest"
)

query_answering_llm = ChatOpenAI()


embeddings = OpenAIEmbeddings(model = "text-embedding-3-small", dimensions=1024)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200  
)

prompt = PromptTemplate(
    template="""
    -Answer the Question Precisely
    -always answer from the given context below
    -if not present in the context, say "Could Not Find Relevent Info in the Context"

    Context: {context}

    Question : {question}

    """, 
    input_variables=["context", "question"]
)


#Runnables
def upload_video(file_path):
    try:
        video = client.files.upload(file=file_path)
        print("Video uploaded, waiting for ACTIVE state...")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

    try:
        while True:
            video = client.files.get(name=video.name)  # refresh status
            if video.state.name == "ACTIVE":
                print("✅ Video is ACTIVE")
                break
            time.sleep(2)
    except Exception as e:
        print(f"❌ Error while checking video status: {e}")
        return None

    return video


runnable_upload_video = RunnableLambda(upload_video)


def prompt_creation(video):
    try:
        if video is None or not hasattr(video, "uri"):
            raise ValueError("Invalid video object passed to prompt_creation")
        
        prompt = [
            HumanMessage(content=[
                {"type": "media", "mime_type": video.mime_type, "file_uri": video.uri},
                {"type": "text", "text": "Please analyze this video in detail. Create a Detailed Description"}
            ])
        ]
        print("Prompt Created")
        return prompt
    except Exception as e:
        print(f"❌ Prompt creation failed: {e}")
        return []

runnable_prompt = RunnableLambda(prompt_creation)


def description_storing(description):
    global vector_store, namespace
    try:

         # generate new namespace every run
        namespace = str(uuid.uuid4())[:8]

        chunks = splitter.split_text(description)
        if not chunks:
            raise ValueError("No chunks generated from description")

        # Store embeddings in Pinecone
        vector_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace,
        )

        print(f"✅ Description Stored in Pinecone (namespace={namespace})")
        return description
    except Exception as e:
        print(f"❌ Storing description failed: {e}")
        return ""

runnable_description_storing = RunnableLambda(description_storing)



def retriever_func(question):
    global vector_store
    try:
        if vector_store is None:
            raise ValueError("Vector store not initialized")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4, "namespace": namespace})
        retrieved_docs = retriever.invoke(question)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print("✅ Context Created")
        return {"context": context_text, "question": question}
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return {"context": "Error retrieving documents", "question": question}



runnable_retrieve_docs = RunnableLambda(retriever_func)
        


parser = StrOutputParser()

#Description Generating & Storing Chain
description_chain = runnable_upload_video | runnable_prompt | description_generating_llm | parser | runnable_description_storing


#Query Answering Chain
QandA_chain = runnable_retrieve_docs | prompt | query_answering_llm | parser






# Export these for API usage
__all__ = [
    "description_chain",
    "QandA_chain"
]

