from langchain_core.pydantic_v1 import BaseModel, Field
import os
import tempfile
import streamlit as st  
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Define structured output model for CV extraction
class CVData(BaseModel):
    skills: list[str] = Field(description="Technical and professional skills from CV")
    work_experience: list[str] = Field(description="Job roles, companies, and durations")
    certifications: list[str] = Field(description="Professional certifications and licenses")

def process_cv(pdf_path):
    # PDF processing and vector store setup
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " "]
    )
    
    chunks = text_splitter.split_documents(pages)
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function, 
        persist_directory="cv_vectorstore"
    )
    vectorstore.persist()
    
    return vectorstore.as_retriever()

def extract_cv_sections(retriever):
    # Enhanced extraction prompt
    PROMPT_TEMPLATE = """
    Analyze the CV and extract structured data. Focus on:
    - Skills: Technical skills, tools, languages (e.g., Python, Project Management)
    - Work Experience: Job titles, companies, dates (e.g., 'Software Engineer @ Google, 2020-2022')
    - Certifications: Professional certificates (e.g., AWS Certified, PMP)
    
    {context}
    """
    
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs))}
        | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        | llm.with_structured_output(CVData, strict=True)
    )
    
    return rag_chain.invoke("Extract key sections")

# Streamlit UI
def main():
    st.title("Internship Matchmaker")
    
    uploaded_file = st.file_uploader("Upload CV (PDF)", type="pdf")
    search_query = st.text_input("Search internships by keywords")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        retriever = process_cv(tmp_path)
        cv_data = extract_cv_sections(retriever)
        
        # Display extracted data
        with st.expander("Extracted CV Data"):
            st.subheader("Skills")
            st.write(", ".join(cv_data.skills))
            
            st.subheader("Work Experience")
            st.write("\n".join(cv_data.work_experience))
            
            st.subheader("Certifications")
            st.write(", ".join(cv_data.certifications))
        
        # Filtering logic (replace with your internship data)
        internships = pd.DataFrame({
            'Title': ['Software Engineer Intern', 'Data Analyst Intern'],
            'Skills': ['Python, SQL, AWS', 'Excel, Tableau, Statistics'],
            'Requirements': ['CS degree, AWS Certified', 'Business Analytics certification']
        })
        
        # Search filter
        if search_query:
            filtered = internships[
                internships['Skills'].str.contains(search_query, case=False) |
                internships['Requirements'].str.contains(search_query, case=False)
            ]
            st.write("Matching Internships:", filtered)
        else:
            st.write("All Internships:", internships)

if __name__ == "__main__":
    main()