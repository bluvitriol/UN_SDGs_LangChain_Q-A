# 🌍 UNDP Generative AI Demo: LangChain + Retrieval Q&A

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain, ChromaDB, and Hugging Face embeddings**, with a smart **fallback between OpenAI and Hugging Face models**.  
The demo loads **UN Sustainable Development Goals (SDGs)** text and enables natural-language Q&A aligned with the UNDP’s mission (reducing poverty, inequality, promoting sustainability).


---
## Features
- **Vector Database (ChromaDB):** Stores SDG documents for retrieval.  
- **Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic search.  
- **LLM Fallback Logic:**
  - Tries **OpenAI GPT-4o-mini** first (if API key + credits available).  
  - Falls back to **Hugging Face Flan-T5** (free Inference API) if OpenAI is unavailable.  
- **Prompt Engineering:** Custom prompt template:  
  > *“You are a UNDP research assistant. Context: {context} … Answer:”*  
- **Retrieval Q&A:** Ask questions like *“Which SDG talks about reducing inequality?”* and get accurate answers.  

---

## Project Structure

**LangChain UN SDGs Q&A/**

``` bash
│── sdg_goals.txt # Dataset (subset of UN SDGs) 
│── rag_fallback.py # Main script (OpenAI → HF fallback)
│── UNDP_GenerativeAI_QA_Brief.pdf # Analytical brief (results & implications)
│── README.md # Documentation
│── requirements.txt # Dependencies
│── .gitignore # Ignore sensitive/junk files
│── .env # API keys (not pushed to GitHub)
```


---

## Getting Started

### 1. Install dependencies
```bash
pip install -qU "langchain[openai]" langchain-community chromadb python-dotenv
```


### 2. Add your API keys

Create a .env file in the project root:
```bash
OPENAI_API_KEY=sk-xxxxxx        # optional, if you still have credits
HUGGINGFACEHUB_API_TOKEN=hf_xxx # free token from Hugging Face
```
### 3. Run the demo
```bash
python rag_fallback.py
```

### 4. Example Output
```bash
OpenAI unavailable or credits exhausted. Falling back to Hugging Face...
Using Hugging Face Flan-T5

Q: Which SDG talks about reducing inequality?
A: Goal 10: Reduce inequality within and among countries.

Q: What is the goal related to climate change?
A: Goal 13: Take urgent action to combat climate change and its impacts.

Q: How does SDG address gender equality?
A: Goal 5: Achieve gender equality and empower all women and girls.
```
---
## Why This Matters

This project showcases:

- **Generative AI for development** (UNDP context).
- **Prompt engineering** + **retrieval chaining** with LangChain.
- **Bias-free fallback strategy** ensuring reliability even with API credit limits.
- **Practical application of LLMs** to global challenges like inequality, gender equality, and climate change.

---
## References

- [LangChain Docs](https://python.langchain.com/docs/introduction/)
- [Hugging Face Models](https://huggingface.co/models)
- [UN Sustainable Development Goals](https://sdgs.un.org/goals)

---
## Next Steps

- Extend dataset to full SDG text corpus.
- Add evaluation scripts for bias and robustness.
- Explore workflow integration with Microsoft Power Automate.
