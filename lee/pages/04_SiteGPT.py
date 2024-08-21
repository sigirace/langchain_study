from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

## 가장 높은 점수를 가진 답변을 고르고, 만약 점수가 같다면 더 최신 답변을 고르도록 지시
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

## Retriever의 output Documents, RunnablePassthrough로 들어온 question을 활용
## Map Reduce 방식을 사용함에 따라, Retriever를 통해 나누어진 각 문서마다(doc)의 점수를 매기기 위해 함수 구현
## 각 doc마다의 답변과 그 source, 일시를 포함해 return에 담는다.
## Return 값은 Dictionary 이며, question(str), answers(dict)로 구성됨.
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
                ## 더 많은 데이터를 담기
            }
            for doc in docs
        ],
    }

## get_answers의 return value가 inputs가 됨 (dict{str, dict})
## condensed 에 answers의 각 값들을 이어붙어(join) single string로 만듬
## choose_prompt에 question(str), answers(str)을 넣어 invoke
## Return 값은 Promptvalue(str) 이다.
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]


    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

## page parser
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()

    if footer:
        footer.decompose()

    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    ## 지나치게 빠른 요청으로 인한 page block 방지
    loader.requests_per_second = 2

    docs = loader.load_and_split(text_splitter = splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

    return vector_store.as_retriever()


st.set_page_config(
    page_title = "SiteGPT",
    page_icon = "🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")

        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                ## Output : {"docs" : List of Document (from Retriever), "question" : query}
                | RunnableLambda(get_answers)
                ## Output : answer_chain의 invoke result(answer, source, result)
                ## 이미 answer_chain에 llm을 걸어놓았으므로 본 chain에서는 llm 없음
                | RunnableLambda(choose_answer)
                ## Output : choose_chain의 invoke result(str)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))