import streamlit as st
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
import whisper
from moviepy.editor import VideoFileClip

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.title("Video to Audio Extractor & Transcription using Whisper and LangChain")

# Step 1: Ask the user to upload a video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file is not None:
    # Step 2: Save the uploaded file temporarily
    video_path = os.path.join("temp_video.mp4")
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Step 3: Load the video file using MoviePy
    video_clip = VideoFileClip(video_path)

    # Step 4: Extract the audio from the video
    audio_clip = video_clip.audio

    # Step 5: Save the extracted audio as an MP3 file
    audio_output = "extracted_audio.mp3"
    audio_clip.write_audiofile(audio_output)

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()
    
    st.success("Audio extracted successfully!")

    # Step 6: Transcribe the extracted audio using Whisper
    st.write("Transcribing the audio, please wait...")

    # Load the Whisper model
    model = whisper.load_model("base")  # You can use other models like "small", "medium", or "large"

    # Transcribe the audio file
    transcription_result = model.transcribe(audio_output)

    # Get the transcribed text
    transcription = transcription_result['text']

    # Step 9: Split the transcription into chunks using LangChain's RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(transcription)

    # Step 10: Generate OpenAI Embeddings for the text chunks
    st.write("Generating embeddings for the transcription...")
    embeddings = OpenAIEmbeddings()
    doc_embeddings = embeddings.embed_documents(docs)

    # Step 11: Store the embeddings in a FAISS vector database
    vector_store = FAISS.from_texts(docs, embeddings)

    # Step 12: Set up the LLM and Prompt using LangChain (Groq or OpenAI)
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    prompt_template = ChatPromptTemplate.from_template("""
    Answer the following question based on the context provided. If the context doesn't contain the answer, say you don't know.
    Context: {context}
    Question: {input}
    """)

    # Step 13: Create the document chain for retrieving information
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Step 14: Use the vector store as a retriever
    retriever = vector_store.as_retriever()

    # Step 15: Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Step 16: Set up user input for asking questions
    user_query = st.text_input("Ask a question about the lecture")

    if user_query:
        # Step 17: Use the retrieval chain to get an answer based on the transcription
        st.write("Processing your query, please wait...")
        response = retrieval_chain.invoke({"input": user_query})
        st.write("**Answer**")
        st.write(response['answer'])

    # Clean up temporary files
    os.remove(video_path)
    os.remove(audio_output)
