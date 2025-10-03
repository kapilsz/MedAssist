import os
import json
from typing import Optional, List, Any

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient

# Custom LLM wrapper for InferenceClient
class HuggingFaceInferenceLLM(LLM):
    client: Any = None
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_tokens: int = 512
    temperature: float = 0.5
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(
            provider="featherless-ai",
            api_key=os.environ["HF_TOKEN"],
        )
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceInferenceLLM(
        model=huggingface_repo_id,
        max_tokens=512,
        temperature=0.5
    )
    return llm

#Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

if __name__ == "__main__":

    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    HF_TOKEN = config.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

    # Load Database
    DB_FAISS_PATH = "vectorstore/db_faiss"
    print("\n" + "="*70)
    print("🔄 Loading Vector Database...")
    print("="*70)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("✅ Database loaded successfully!")

    # Create QA chain
    print("\n🤖 Initializing AI Model (Mistral-7B-Instruct)...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    print("✅ Model initialized successfully!")

    # Now invoke with a single query
    print("\n" + "="*70)
    user_query = input("💬 Write Query Here: ")
    print("="*70)
    print("\n⏳ Processing your query...\n")

    response = qa_chain.invoke({'query': user_query})

    print("\n" + "="*70)
    print("📝 ANSWER:")
    print("="*70)
    print(response["result"])

    print("\n" + "="*70)
    print("📚 SOURCE DOCUMENTS:")
    print("="*70)
    for i, doc in enumerate(response["source_documents"], 1):
        print(f"\n📄 Document {i}:")
        print("-" * 70)
        content = doc.page_content
        # Show first 300 characters of each document
        if len(content) > 300:
            print(content[:300] + "...\n")
        else:
            print(content + "\n")
        
        # Show metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"📌 Source: {doc.metadata}")

    print("\n" + "="*70)