import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

prompt_template = """
You are a friendly and helpful cooking assistant. Based on the recipe context provided below, please answer the user's question.

Your answer must be structured into three parts:
1. Start with a brief, friendly introduction or 'bridging words'.
2. List the ingredients clearly under the heading '### Ingredients'.
3. Provide the step-by-step instructions under the heading '### Instructions'.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

@st.cache_resource
def load_and_process_data():
    """
    Loads recipes, creates embeddings, and sets up the RAG chain.
    """
    print("Loading and processing data... (This will run only once)")
    loader = JSONLoader(
        file_path='./recipes.jsonl',
        jq_schema='.content',
        text_content=False,
        json_lines=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain

# --- Main Streamlit App ---

st.title("🍳 Recipe Assistant Chatbot")
st.write("Ask me any question about the recipes I have, and I'll find the answer for you!")

try:
    qa_chain = load_and_process_data()

    user_question = st.text_input("What would you like to cook today?")

    if st.button("Get Recipe"):
        if user_question:
            with st.spinner("Finding the perfect recipe..."):
                # --- NEW: Check if any relevant documents are found FIRST ---
                retriever = qa_chain.retriever
                retrieved_docs = retriever.invoke(user_question)

                if not retrieved_docs:
                    # If the retriever returns an empty list, show a custom message
                    st.warning("I'm sorry, I couldn't find that recipe in my cookbook. Please try another one!")
                else:
                    # If documents are found, proceed with the LLM call as before
                    response = qa_chain.invoke(user_question)
                    response_text = response['result']

                    try:
                        parts = response_text.split('### Ingredients')
                        bridging_words = parts[0].strip()
                        
                        ing_and_inst = parts[1].split('### Instructions')
                        ingredients = ing_and_inst[0].strip()
                        instructions = ing_and_inst[1].strip()

                        st.markdown(bridging_words)
                        st.subheader("Ingredients")
                        st.markdown(ingredients)
                        st.subheader("Instructions")
                        st.markdown(instructions)

                    except (IndexError, ValueError):
                        st.warning("The model provided a response, but it couldn't be automatically structured. Here is the raw output:")
                        st.write(response_text)
        else:
            st.warning("Please ask a question first!")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please make sure your GOOGLE_API_KEY is set correctly in the .env file and that all dependencies are installed.")