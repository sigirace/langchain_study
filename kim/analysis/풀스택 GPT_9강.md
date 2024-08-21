# 9 QuizGPT


## 9.0 Introduction

Quiz PT는 파일을 받는다는 점에서 document GPT와 좀 비슷하지만 파일에 대한 질문을 하는 대신에 
LLM이 우리에게 파일의 내용과 관련된 질문을 하도록 명령할 것임.

Wikipedia Retriever를 사용 : 위키피디아에서 본문 가져오고 LLM이 가져온 본문기반 질문

이 섹션과 Quiz GPT의 중요 포인트는 output parser를 배우는 것임 : 모델이 특정형식으로 답을 하도록 강제하기

LangChain의 output parser를 통해 퀴즈의 정답을 파싱: Function Calling

Function Calling을 이용하면 우리가 만든 함수가 어떻게 생겼는지, 어떤 parameter 값을 원하는지 LLM에게 설명할 수 있다.
그 뒤 우리가 LLM에게 질문을 했을 때 LLM이 text로 답하는게 아니라 우리가 작성한 함수들을 호출하게 된다. => 핵심기능!


## 9.1 WikipediaRetriever

사용자의 파일시스템으로부터 파일을 받아서 split 한 문서를 생성 : 

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path) 
    docs = loader.load_and_split(text_splitter=splitter)

Wikipedia Retriever를 사용해서 관련된 문서를 가져오기 : retriever = WikipediaRetriever(top_k_results=5)

## 9.2 GPT-4 Turbo

> 생성한 텍스트 문서를 어떻게 LLM에 전달할 것인가? > 그리고 그것에 대해 몇가지 질문을 생성하도록 만들기

> 새로운 버전의 GPT3, GPT4는 엄청나게 큰 context window를 가진다.(Models - OpenAI API : GPT-4 Turbo and GPT-4는 무려 128,000 tokens을 context window로 가진다. (하나의 프롬프트 안에 300 페이지 정도의 문자가 있는것과 같음)

> 가격정책(Pricing | OpenAI)을 보면 gpt-4-1106-preview $10.00 / 1M tokens $30.00 / 1M tokens로 나쁘지 않다. > 쳅터 전체를 하나의 프롬프트에 담아서 보낼 수가 있다!

> gpt-4-1106을 사용하는 방법
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
)

 
## 9.3 Questions Prompt
llm에게 문서와 관련된 질문을 물어봐 달라고 하기 > 그 답변을 우리가 사용할 수 있는 형태로 가공하기

from langchain.prompts import ChatPromptTemplate

Model의 실시간 답변을 출력해주는 아래 모듈 imort
from langchain.callbacks import StreamingStdOutCallbackHandler

>> 모든 작업이 다 끝나기 전에 Quiz가 어떻게 생성되고 있는지 볼 수 있게 되고 실수가 있었거나 수정할 일이 생기면 Model을 멈출 수도 있다. > llm정의에 아래 인자를 추가

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming = True,
    callbacks = [StreamingStdOutCallbackHandler()],
)

>> 다음엔 chain을 만든다.

   chain = {
        "context": format_docs
    } | prompt | llm
    
    start = st.button("Generate Quiz")
    
    if start:
        #chain.invoke({"context": })
        chain.invoke(docs)
>> 우리가 가진 모든 문서를 chain을 불러올 때 넘겨주고 그 문서는 format_docs로 넘어간 뒤에 format_docs는 string을 반환하고 그것이 context값이 될 것이다. 그리고 context는 그걸 필요로 하는 prompt 안으로 들어가게 된다.

>> 답변을 포맷팅하기 위한 또다른 모델이 필요할 수도 있다.


## 9.4 Formatter Prompt

체인을 다시 만들어서 선생님 GPT로부터 받은 정보를 Formatting
>> 코드를 정리하여 Streamlit 로직과 LLM 로직을 분리해서 작성해보자 > prompt 부분과 chain을 위로 올려서 이름을
questions_prompt, questions_chain으로 바꾸기

>> formatting chain을 만들어보자 : 만들어진 퀴즈 문제를 받아서 json처럼 format해줄 system prompt 만들기, 그렇게 하면 파이썬에서 더욱 빠르게 처리해서 UI 씌우기 간편

>> formatting_prompt에서 "{{"를 넣은 이유는 langchain이 이 부분은 format 하지 않길 바라기 때문이다. 어떤 값을 주입할 때 중괄호를 사용하는데 ({context}와 같이) 중괄호를 두개 넣음으로써 그러지 않게 함.

Streamlit 로직에 아래와 같이 셋팅

        questions_response = questions_chain.invoke(docs)
        st.wrtie(questions_response.content)
        formatting_response = formatting_chain({
            "context": questions_response.content
        })
        st.write(formatting_response.content)

>> .content를 사용하는 이유는 기본적으로 우리가 체인에서 AI Message에서 받아오게 되므로 AI Message안에 content가 있다. 우리는 그것이 필요한 것이다. 그 외에 값은 신경쓰지 않는다.

>> formatting_prompt을 추가함으로써 json 형태의 답안지를 받아볼 수 있다.


## 9.5 Output Parser 
위 값들을 이용하여 실제 UI를 입혀보기 > LLM에서 가져온 JSON 또는 JSON 문자열을 사용할 수 있도록 output parser를 만들어보자


       # questions_response = questions_chain.invoke(docs)
       # st.write(questions_response.content)
       # formatting_response = formatting_chain.invoke({
       #     "context": questions_response.content
       # })
       # st.write(formatting_response.content)
       

       chain = {"context": questions_chain} | formatting_chain
        
>> 위 코드 대신에 아래 체인을 document와 함께 호출한다면 그 documents 들은 questions_chains으로 전달될 거고 우리 document에 대한 질문을 받아올 거고 그것이 context로 전달되고 그 context는 formatting_chain으로 전달된다.

>> chain에 output_parser을 추가하기 위해 아래와 같이 클래스를 추가한다. : LLM에 의해 생성된 text를 받는 parse 함수 생성된 클래스

from langchain.schema import BaseOutputParser

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        return super().parse(text)

>> 우리의 LLM이 ```json  ~~~ ```을 포함하고 있으므로 이걸 텍스트에서 제거하고 JSON 모듈을 사용하여  JSON 처럼 생긴 문자열을 우리의 파이썬 코드에서 사용할 수 있는 실제 object로 만들면 된다.

>> 아래과 같이 우리의 output parser을 정의

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()

>> chain 정의 이후 구문이 아래와 같이 변경
  chain = {"context": questions_chain} | formatting_chain | output_parser
  response = chain.invoke(docs)
  st.write(response)


>> 실행해보면 실제 우리가 사용할 수 있는 파이썬 오브젝트 형태로 코드가 리턴됨


## 9.6 Caching
### 9.5에서 리턴된 object를 매우 아름다운 form 안에 넣어주는 로직을 학습

>> streamlit은 무언가 변경될 때 모든 코드를 실행함 > 아래 코드는 다시 실행하는데 시간이 걸리기 때문에 캐싱을 하고 싶음(document 들이 바뀌지 않는다면 아래 코드가 다시 실행되지 않도록)

        chain = {"context": questions_chain} | formatting_chain | output_parser
        response = chain.invoke(docs)


>> 캐시함수 생성시 해시할 수 없는 매개변수가 있거나 streamlit이 데이터의 서명을 만들 수 없는 경우 다른 매개변수를 추가해서 이것이 변경되면 streamlit이 함수를 재실행 시킬 수 있도록 처리

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


## 9.7 Grading Questions
### UI 만들어보기


## 9.8 Function Calling 
### 모델이 우리의 코드를 호출하도록 조정 or 모델이 우리가 원하는 특정 모양과 형식의 output을 갖도록 강제
>> 함수호출은 LLama와 같은 무료모델 사용시 사용불가하고 GPT3, GPT4 에서만 가능

>> 어떻게 LLM이 get_weather이라는 함수가 있다는 걸 알도록 할까? > 우리가 해야하는 건 스키마(schema)를 만드는 것이다.(이 함수를 위한 JSON Schema) > 스키마를 생성해준다 : 이 함수의 작동 방식과 필요한 것을 설명해주는 스키마

def get_weather(lon, lat):
    print("call an api ... ")
    
Function = {
    "name": "get_weather",
    "description": "function that takes longitude and latitude to find the weather of a place",
    "parameters": {
        "type": "object",
        "properties": {
            "lon": {
                "type": "float",
                "description": "The longitude coordinate"},
            "lat": {
                "type": "float",
                "description": "The latitude coordinate"},
            },
        },
    "required": ["lon", "lat"],
}

>> 위 함수를 ChatOpenAI가 사용할 수 있도록 하려면? 

llm = ChatOpenAI(
    temperature= 0.1,
).bind(
    #function_call={"name": "get_weather"}, # 모델이 강제로 함수를 사용하도록
    function_call="auto", # 모델이 필요에 따라 함소호출여부를 선택하여 사용하도록
    functions=[
        function
    ]
)

>> functions에 원한다면 많은 함수를 넣어줄 수 있다. 그리고 function_call에는 기본적으로 모델이 특정 함수를 사용하도록 강제하거나 모델이 함수를 사용하도록 하거나 그냥 답변을 할 수 있도록 모델 스스로 선택하게 할 수 있다.

>> 이전에 공부했던 답변을 위해 한 번, format을 위해 또 한번 호출할 필요가 없이 한번의 llm 호출로 이렇게 만듬.
> gpt-3, gpt-4에서만 가능하고 다른 모델(ollama)에서는 작동하지 않으므로 output parser와 함수 호출, 두가지 옵션을 보여준 것이다. > 함수를 작성할 것인가, 프롬프틀 통해 예제를 보여줄 것인가는 내가 하고 싶은게 뭐냐에 따라 달렸음

## 9.9 Conclusions