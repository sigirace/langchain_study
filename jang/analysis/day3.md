# 6.0 Introduction

- RAG(Retrieval Argumented Generator)
- model들은 학습은 하지만 개인적인 데이터는 접근 불가하여 고안된 방법
- llm은 학습된 모델 + 사용자가 주어주는 데이터까지 반영하여 답
- Retrieval은 langchain의 모듈
- source>load>transform>embed>store>return
- chunk_size를 통해 문단을 쪼개기
- chunk_overlap을 통해 문단 잘리는걸 방지(문장의 끝의 중복허용)
- 청크는 작을수록 좋다
- CharacterTextSplitter가 separator가 있어서 자주 사용함

# 6.2 Tiktoken

- Token!=length

# 6.3 Vectors

- embedding : 텍스트를 컴퓨터가 이해가능한 언어(벡터)로 바꾸는 작업
- https://turbomaze.github.io/word2vecjson/
- https://www.youtube.com/watch?v=2eWuYf-aZE4

# 6.4 Vector Store

- openAI 는 1536D네
- 테스트할때는 langchain.storage.localfilestore로 캐싱해서 테스트할수있겟다

# 6.5 Langsmith

- chain의 동작을 시각적으로 표현

# 6.6 RetrievalQA

- retriever은 cloud든 document에서 가져올때 사욯하는 인터페이스/class이다. doc을 검색해서 찾아오는 기능을 담당
- stuff는 문서를 prompt에 채워넣는거래
- refine은 질문과 관련된 내용을 doc에서 찾고 답변을 생성하는 일련의과정 반복하며 답변을 정제,doc의 숫자만큼 질문해야함
- map reduce : doc을 입력받아 요약작업후 llm에게 전달,큰 연산작업이 일어남
- map re-rank : doc을 입력받아 답변을 생성후 답변의 점수를 매겨 높은애를 답변
- 할루시네이션 방지를 위해 모르면 모른다고 말하라는 prompt 필수

# 6.8 Stuff LCEL Chain

- retriever's input: single string,질문 혹은 doc을 얻기위한 쿼리
- 동작순서
  - 질문을 retriever에 전달
  - 그결과(doc)를 prompt context에 전달
  - 질문을 prompt의 question에 전달
  - 최종 질의를 llm에 전달
- RunnablePassthrough : 데이터를 변경하지 않고 파이프라인의 다음 단계로 전달하는 데 사용
  - 데이터를 변환하거나 수정할 필요가 없는 경우
  - 파이프라인의 특정 단계를 건너뛰어야 하는 경우
  - 디버깅 또는 테스트 목적으로 데이터 흐름을 모니터링해야 하는 경우

# 6.9 Map Reduce LCEL Chain

- retriever가 검색결과로 doc이 크면 stuff 사용불가
- 회의 요약하는 등 모든 doc을 다읽어서 처리해야할 경우 순차적으로 llm에 보내는 기능 구현
- RunnableLambda는 chain과 그 내부 어디서든 function을 호출 가능하게 함

# 7.0 Introduction

# 7.1 Magic

- st.write(), 변수/class등을 지가 알아서 보여줌

# 7.2 Data Flow

- streamlit은 변경시마다 모든 데이터를 리로드함(단순 selectbox 데이터 바뀔떄도 리로드)

# 7.3 Multi Page

- with로 코드량을 줄일 수 있다

# 7.4 Chat Messages

- 챗봇을 구현하려면 리로드되어 날아가는 데이터들을 저장해둘곳이 필요

# 7.6 Uploading Documents

- streamlit 특성상 데이터변경시마다 리로드하므로 cache 기능이 중요

# 7.7 Chat History

- cache data decorator로 file이 동일할때 기존 캐싱값을 리턴

# 7.8 Chain

- chain을 사용하면 아래 일련의 과정들을 내부적으로 수행해줌
  - retriever.invoke해서 관련 문서 서칭
  - 찾은 문서를 prompt에 삽입
  - 최종 질의 llm전달

# 7.9 Streaming
