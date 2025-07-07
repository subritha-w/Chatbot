import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variable
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit layout
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.markdown("<h2 style='text-align: center;'>📄 PDF Chatbot using Gemini</h2>", unsafe_allow_html=True)

# Upload PDF
pdf = st.file_uploader("Upload your PDF file", type="pdf")

if pdf:
    with st.spinner("Processing..."):
        with open("uploaded_file.pdf", "wb") as f:
            f.write(pdf.read())

        # Load and split
        loader = PyMuPDFLoader("uploaded_file.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = splitter.split_documents(documents)

        # Debug: Show how many chunks
        st.success(f"✅ PDF loaded. Chunks created: {len(docs)}")
        if len(docs) > 0:
            st.code(docs[0].page_content[:300], language='text')

        if len(docs) == 0:
            st.error("⚠️ No content found in the uploaded PDF. Please try a different file.")
            st.stop()

        # Embeddings + VectorStore
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # Memory (FIXED)
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        # Chain (FIXED)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory,
            return_source_documents=True,
            output_key="answer"
        )

        # Chat history
        if "chat" not in st.session_state:
            st.session_state.chat = []

        # User input
        query = st.chat_input("Ask your question about the PDF...")

        if query:
            st.session_state.chat.append(("user", query))

            # Simple small talk
            def small_talk(query):
                q = query.lower().strip()
                if q in ["hi", "hello"]:
                    return "👋 Hi there!"
                elif "thank" in q:
                    return "You're welcome! 😊"
                elif "bye" in q:
                    return "Goodbye! 👋"
                elif "how are you" in q:
                    return "I'm just a bot, but I'm happy to help!"
                return None

            with st.spinner("Thinking..."):
                answer = small_talk(query)
                if not answer:
                    result = qa_chain.invoke({"question": query})
                    answer = result["answer"]

                    # Gemini fallback message detection
                    if "i do not have access to files" in answer.lower():
                        answer = "⚠️ It seems the answer could not be retrieved. Try rephrasing your question or upload a clearer PDF."

            st.session_state.chat.append(("ai", answer))

        # Display chat
        for role, msg in st.session_state.chat:
            with st.chat_message(role):
                st.markdown(msg)
