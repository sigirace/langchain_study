from langchain.chat_models import ChatOpenAI
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import whisper
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS 
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings


# llm = ChatOpenAI(
#     temperature=0.1,
    
# )

llm = ChatOllama(model="llama3.2:latest")

has_transcript = os.path.exists("./.cache/test.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        # model="chatfire/bge-m3:q8_0" # BGE-M3
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


# audio를 읽어들여 txt파일로 변환
@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return

    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()

    model = whisper.load_model("base")
    final_transcript = ""

    ## 무료버전
    for file in files:
        transcript = model.transcribe(
            file
        )  # with문으로 as(알리아스) 변수를 사용하면 ndarray 오류가 나므로 파일 바로 사용
        final_transcript += transcript["text"]

    with open(destination, "a") as text_file:
        text_file.write(transcript["text"])

    # ## 유료버전
    # for file in files:
    #     # print(file)
    #     with open(file, "rb") as audio_file, open(destination, "a") as text_file:
    #         transcript = openai.Audio.transcribe("whisper-1", audio_file)
    #         text_file.write(transcript["text"])


@st.cache_data()
def cut_audio_in_chunks(autio_path, chunk_size, chunks_folder):
    if has_transcript:
        return

    track = AudioSegment.from_file(autio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]

        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return

    # audio_path = video_path.replace("mp4", "mp3")
    audio_path = video_path[:-4] + '.mp3'
    # ffmpeg -i files/podcast.mp4 -vn files/audio.mp3
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # MeetingGPT
            
    Welcome to MeetingGPT, upload a video and I will give you a transcript, a
    summary and a chat bot to ask any questions about it.
    
    Get started by uploading a video file in the sidebar.         
"""
)


with st.sidebar:
    video = st.file_uploader("Audio", type=["mp4", "avi", "mkv", "mov", "m4a"])

if video:
    chunks_folder = "./.cache/chunks"

    with st.status("Loading video...") as status:

        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        # audio_path = video_path.replace("mp4", "mp3")
        audio_path = video_path[:-4] + '.mp3'
        # transcript_path = video_path.replace("mp4", "txt")
        transcript_path = video_path[:-4] + '.txt'
        
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)

        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 2, chunks_folder)

        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])
    
    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
            
    with summary_tab:
        start = st.button("Generate summary")
        
        if start:
            # summary template 
            loader = TextLoader(transcript_path)
            
            docs = loader.load_and_split(text_splitter=splitter)
            
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                다음의 내용을 간략하게 요약합니다.
                "{text}"
                간략한 요약: 
                """
            )
            
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            
            summary = first_summary_chain.invoke({
                "text": docs[0].page_content
            })
            
            refine_prompt = ChatPromptTemplate.from_template(
                """
                당신의 일은 최종 요약을 만드는 것입니다.
                우리는 특정 지점까지의 기존 요약을 제공했습니다 : {existing_summary}
                필요한 경우에는 아래에 더 제공되는 맥락(context)으로 기존 요약을 다듬어도 됩니다.
                ------------
                {context}
                ------------
                주어진 새 context를 통해, 기존 요약을 다듬어 주세요.
                만약, context가 유용하지 않다면 기존 요약을 반환해주세요.
                """
            )
            
            refine_chain = refine_prompt | llm | StrOutputParser()
            
            with st.status("Summarizing...") as status:
                # 첫번째 문서를 만들고 시작하므로 첫번째는 건너뜀 
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke({
                        "existing_summary": summary,
                        "context": doc.page_content,
                    })
                
                    st.write(summary) # 중간 요약 추적용 
                    
                st.write(summary)
    
    with qa_tab:
        retriever = embed_file(transcript_path)
        
        docs = retriever.invoke("HI")
        
        st.write(docs)