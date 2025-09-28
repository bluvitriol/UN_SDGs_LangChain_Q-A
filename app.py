import os
import sys
import streamlit as st
from dotenv import load_dotenv

#SQLite fix for Chroma
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="UN SDG Q&A Assistant", page_icon="🌍")

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#retriever
@st.cache_resource(show_spinner="Building or loading retriever...")
def build_retriever():
    persist_dir = "chroma_store"

    loader = TextLoader("sdg_goals.txt")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)

    return db.as_retriever()


retriever = build_retriever()

#prompt
prompt = PromptTemplate(
    template=(
        "You are a UNDP research assistant. "
        "Use the following context to answer clearly and concisely.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    input_variables=["context", "question"],
)

def init_qa():
    """Initialize QA with OpenAI if available, else Hugging Face."""

    qa = None
    # 1)trying openai api
    try:
        from langchain.chat_models import init_chat_model

        if openai_key:
            llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
            _ = llm.invoke("ping") #quick test
            qa = RetrievalQA.from_chain_type(
                llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
            )
            st.sidebar.success("Using OpenAI GPT-4o-mini")
            return qa
        else:
            raise RuntimeError("OPENAI_API_KEY not set.")
    except Exception as e_open:
        st.sidebar.warning(f"OpenAI unavailable: {str(e_open)}")

    # 2)Hugging Face fallbacks
    from langchain_community.llms import HuggingFaceEndpoint

    hf_candidates = [
        # repo_id, task, temperature, max_new_tokens
        ("google/flan-t5-base", "text2text-generation", 0.0, 256),
        ("google/flan-t5-small", "text2text-generation", 0.0, 128),
        ("tiiuae/falcon-7b-instruct", "text-generation", 0.3, 256),
    ]

    for repo_id, task, temperature, max_new_tokens in hf_candidates:
        try:
            if not hf_token:
                raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set.")

            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                huggingfacehub_api_token=hf_token,
                task=task,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            # quick smoke test
            _ = qa.invoke({"query": "Which SDG talks about reducing inequality?"})
            st.sidebar.success(f"Using Hugging Face fallback: {repo_id}")
            return qa
        except Exception as e_hf:
            st.sidebar.warning(f"{repo_id} failed: {str(e_hf)}")
            continue


    st.sidebar.error("No free LLM available")
    return None


qa = init_qa()

#UI
st.title("🌍 UN SDG Q&A Assistant")
st.write(
    "Ask me questions about the **Sustainable Development Goals (SDGs)**. "
    "I’ll retrieve context from UN documents and provide clear answers."
)

user_query = st.text_input("Enter your question:")
if user_query:
    if qa:
        with st.spinner("Thinking..."):
            response = qa.invoke({"query": user_query})
        st.write("### Answer")
        st.success(response["result"])
    else:
        st.error("No language model is available. Please check your keys/settings.")

#sidebar text
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app is powered by **LangChain**, **Chroma**, and **LLMs (OpenAI/HuggingFace)**.\n\n"
    "It was built as a demo project for **UNDP-style SDG Q&A**."
)

st.sidebar.markdown("#### *Note:*")
st.sidebar.info(
    "To ask another question, just type in the question and press **Enter**."
)
