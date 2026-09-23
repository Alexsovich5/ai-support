"""
AI Engine - LangChain RAG Pipeline
"""

import logging
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

logger = logging.getLogger("ai_engine")


class AIEngine:
    def __init__(self, config):
        self.llm = ChatOpenAI(model_name=config.get("model", "gpt-4"), temperature=0.1)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=config.get("chroma_path", "./data/chroma"),
            embedding_function=self.embeddings,
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
        )

    async def query(self, question: str) -> dict:
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result.get("source_documents", [])],
        }

    async def classify_ticket(self, description: str) -> dict:
        prompt = f"Classify this IT support ticket into one category (network, hardware, software, access, email, other) and priority (P1-Critical, P2-High, P3-Medium, P4-Low):\n\n{description}"
        response = self.llm.predict(prompt)
        return {"classification": response}
