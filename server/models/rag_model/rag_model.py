import os
import warnings
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

warnings.filterwarnings('ignore')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


class AstroAssistant:
    def __init__(self, file_path, model_name="gemma3:1b"):
        self.vector_db = self._create_vector_db(file_path)
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        self.llm = ChatOllama(model=model_name)
        self.chat_history = []
        self.prompt = self._create_prompt()
        self.chain = self._create_chain()

    def _create_vector_db(self, file_path):
        loader = TextLoader(file_path=file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        return Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
            collection_name="local-rag-conversational"
        )

    def _format_docs(self, retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    def _format_history(self):
        if not self.chat_history:
            return "No previous conversation history."
        formatted_history = ""
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                formatted_history += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"Assistant: {message.content}\n\n"
        return formatted_history.strip()

    def _create_prompt(self):
        return PromptTemplate(
            template="""
[SYSTEM PROTOCOL: MULTIMODAL RAG ASSISTANT]
[TASK: Generate a helpful, professional response based on provided CONTEXT and CHAT HISTORY. 
If input contains emotions, acknowledge them appropriately.]

[CONVERSATION HISTORY]
{chat_history}

[CURRENT INPUT DATA]
Retrieved Context: {context}
User Input: {question}

[BEGIN RESPONSE]
""",
            input_variables=['chat_history', 'context', 'question']
        )

    def _create_chain(self):
        retrieval_chain = RunnableParallel({
            'context': RunnablePassthrough() | self.retriever | self._format_docs,
            'question': RunnablePassthrough(),
            'chat_history': RunnableLambda(lambda _: self._format_history())
        })
        parser = StrOutputParser()
        return retrieval_chain | self.prompt | self.llm | parser

    def get_response(self, user_input: str):
        if not user_input:
            return {"response_text": "Input cannot be empty."}
        response = self.chain.invoke(user_input)
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        return {"response_text": response}