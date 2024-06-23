import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from main import detect_distraction 
import cv2

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

questions = []  # List to store questions globally

def save_questions(filename, questions):
    project_dir = os.getcwd()  # Get the current working directory (your project directory)
    filepath = os.path.join(project_dir, filename)  # Create the file path
    with open(filepath, "w") as f:
        for question in questions:
            f.write(question + "\n")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Act as a teacher and answer the question asked by the user.The answer to the question must be from the pdf uploaded by the person.Below the answer i want you to explain the concept in simple manner make sure that the user understands it, and also act as a conversational teacher. Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, frame):  # Removed 'questions' as an argument
    global questions  # Access the global 'questions' list
    # Detect distraction
    distracted = detect_distraction(frame)

    # Proceed with answering only if not distracted
    if distracted:
        print("Sorry, distracted. Cannot answer at the moment.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

    # Save the question to the file
    questions.append(user_question)

    return questions  # Return updated list of questions


def main():
    global questions  # Access the global 'questions' list
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Ask question from the PDF
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        # Initialize video capture
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect distraction
            distracted = detect_distraction(frame)

            # Display result and play sound if distracted
            if distracted:
                print("Sorry, distracted. Cannot answer at the moment.")
                break
            else:
                # Answer the question if not distracted
                questions = user_input(user_question, frame)
                save_questions("C:\\Users\\Shruthi\\OneDrive\\Desktop\\Langchain\\saved_questions.txt", questions)  # Save questions here
                break  # Exit loop after answering the question

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Sidebar for additional functionality
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()


