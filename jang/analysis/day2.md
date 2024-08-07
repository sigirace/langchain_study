# 4.0

- Model I/O : 언어 모델과의 인터페이스(promt+language model+output parser),아래
- retrieval : 외부데이터를 모델에 제공하는 방법
- chain : modelIO랑 뭐가 다름
- agent : chain이 필요한 도구들 제공
- memory :
- callback : 요청에 대해 모델이 어떤작업을 하는지 체크

#### LangChain의 구조

- Prompts : 초거대 언어모델에게 지시하는 명령문
  - Prompt Templates, Chat Prompt Template, Example Selectors, Output Parsers
- Index : LLM이 문서를 쉽게 탐색할 수 있도록 구조화 하는 모듈
  - Document Loaders, Text Splitters, Vectorstores, Retrievers
- Memory : 채팅 이력을 기억하도록 하여 이를 기반으로 대화가 가능하도록 하는 모듈
  - ConversationBufferMemory, Entity Memory, Conversation Knowledge Graph Memory
- Chain : LLM 사슬을 형성하여 연속적인 LLM 호출이 가능하도록 하는 핵심 구성 요소
  - LLM Chain, Question Answering, Summarization, Retrival Question/Answering
- Agents : LLM이 기존 Prompt Template으로 수행할 수 없는 작업을 가능케 하는 모듈
  - Custom Agent, Custom MultiAction Agent, Conversation Agent

# 4.1 fewShotPromptTemplate

- prompt template를 사용하는이유 : 디스크에 저장하고 load하는 일련의 과정을 알아서 처리해줌
- FewShot : llm이 어떻게 대답해야 할 지를 예시(형식화)를 줌
- system message는 어떻게 넣지?

# 4.2 fewShotChatEmssagePromptTemplate

- AI에 answer을 넣어주면서 llm에 이런 히스토리가 있다 라고 속이는 원리

# 4.3 LengthBasedExampleSelector

- example이 많을때 유동적으로 선택하는 내용,비용문제
- maxLength로 자를거면 그냥 example수를 제한하면되잖아

# 4.4 Serializtion and Composition

- Serializtion : 불러오기/저장
- Composition : promptTemplate를 이용해 결합하는

# 4.5 Caching

- 캐싱을 통해 llm의 답변을저장 할 수 있음.동일질문에 대해 재사용
- set_debug : 캐싱에 관련된로그를 띄워줌/ prompt가 뭔지, 모델이름등 을 보여줌/ chain에서 유용한기능
- SQLiteCache : DB에 캐싱

# 4.6 Serializaion

# 5.0 ConversationBufferMemory

- memory는 5종류가 있음,API는 모두 동일
- text completion, 예측,자동완성등에 유용

# 5.1 ConversationBufferWindowMemory

- ConversationBufferMemory : 대화가 길어질수록 메모리가 증가하는 비효율, 제일 간단함,중복도 허용하는거 같은데
- 대화의특정 부분만을저장,선입선출형태,k 파라미터로 대화저장갯수 지정
- 최근 대화만 반영하여 답하는 단점

# 5.2 ConversationSummaryMemory

- 메세지를 그대로 저장하는게 아니라 대화를 요약해서 저장
- 긴대화의 경우 유용,초반에는 다른 메모리저장보다 더많은 용량사용(기능설명등의 내용으로 초반에는 많은 메모리 필요)
- 메모리를 실행하는데도 돈이 든다고함

# 5.3 ConversationSummaryBufferMemory

- Summary + buffer 메모리 조합
- 메모리에 보내온 메세지수를 저장,limit에 도착하면 오래된 메세지들은 요약함
- 메세지의 토큰을 기준으로하고, system message에 요약하네

# 5.4 ConversationKGMemory

- knowledge graph를 그린다는데, 중요한 내용들만 뽑아 요약
- knowledge graph에서 히스토리가 아닌 엔티티를 가져옴

# 5.5 Memory on LLMChain

- 우리의 memory를 chain에 전달하는 내용(memory에만 저장되고 llm에 전달안되면 의미가 없음)
- llm chain is off-the-shelf chain, that means general purpose chain

# 5.6 Chat Based Memory

- 누가 보냈는지 모르거나,메세지 양등이 예측하기 어려울떄 Placeholder 통해 대체

# 5.7 LCEL Based Memory

- 체인을 만드는데 메모리랑 프롬프트가 안바뀐데
- invoke시 chat_history를 계속 주입해줘야하는데 RunnablePassthrough를 통한 체인으로 해결

# 5.8 Recap
