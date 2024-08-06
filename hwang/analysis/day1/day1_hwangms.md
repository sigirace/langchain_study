# 2. Introduction

### 1. Langchain
- LLM을 활용한 어플리케이션을 만들기 위한 framework
- 도큐먼트/임베딩/프롬프트/LLM/Chat 등 여러가지 모듈이 있음

### 2. Streamlit
- UI를 개발하기 위해 Streamlit을 활용할 예정 

### 3. Vector DB
- Pinecone 사용할 예정 (무료)

### 4. API 서버
- FastAPI 사용할 예정

### 5. LLM
- ChatGPT 3.5 예정 (비용이 저렴)



# 환경설정
- python version : 3.11
- venv를 통한 가상환경 사용 


# 3. WELCOME TO LANGCHAIN
## 3.0 LLMs and Chat Models
- 주피터 노트북을 통해 OPEN AI 호출해보기
- GPT 3.5 turbo는 deprecated되어 올라마로 대체함
- temperature : 수치가 높을수록 창의적인 답변을 제공 
- predict : 가장 기본적인 답변 모델 

```python
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.chat_models import ChatOllama

llm = ChatOllama(model="llama3:latest")

# llm = OpenAI()
# chat = ChatOpenAI()

a = llm.predict("How many planets are there?")
# b = chat.predict("How many planets are there?")

a
```
```
결과 : The number of planets in our universe is still a topic of ongoing research and debate. As of now, here's what we know:\n\n**In our solar system:** There are 8 planets officially recognized by the International Astronomical Union (IAU):\n\n1. Mercury\n2. Venus\n3. Earth\n4. Mars\n5. Jupiter\n6. Saturn\n7. Uranus\n8. Neptune\n\n**Exoplanets:** Beyond our solar system, there are thousands of exoplanets that have been discovered so far. According to NASA's Exoplanet Archive, as of March 2023:\n\n1. There are over 4,100 confirmed exoplanets.\n2. Another 3,000+ exoplanet candidates are waiting for confirmation.\n\n**Total:** If we combine the number of planets in our solar system with the estimated number of exoplanets, we're looking at around **5,100 to 7,100 planets**, depending on how you count the exoplanets (some might be considered dwarf planets or other types of celestial bodies).\n\nKeep in mind that new discoveries are being made regularly, so these numbers will likely continue to grow as our understanding of the universe expands!
```

## 3.1 Predict Messages
- SystemMessage : 우리가 LLM에 설정들을 제공하기 위한 메시지
- AIMessage : AI에 의해 보내지는 메시지 
- HumanMessage : 우리가 알고 있는 메시지 (질의)

```python
from langchain.schema import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a geography expert. And you only reply in Italian.",
    ),
    AIMessage(content="Ciao, mi chiamo Paolo!"),
    HumanMessage(content="What is the distance between Mexico and Thailand. Also, What is your name?",)
]

chat.predict_messages(messages)
```

- 아래와 같이 placeholder를 사용할 수도 있음. 이 내용은 다음 강의시간에 배울 예정 
```python
from langchain.schema import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a geography expert. And you only reply in {language}.",
    ),
    AIMessage(content="Ciao, mi chiamo {name}!"),
    HumanMessage(content="What is the distance between {country_a} and {country_b}. Also, What is your name?",)
]

chat.predict_messages(messages)
```

```
result : 
AIMessage(content='Ciao Paolo!\n\nIl mio nome è Geografia, e sono felice di aiutarti! La distanza tra Messico e Tailandia è di circa 18.000 chilometri.\n\nBuona giornata!')
```

## 3.2 Prompt Templates
- prompt의 성능이 좋아야 좋은 답변을 얻을 수 있음 
- 랭체인 프레임워크의 큰 부분을 프롬프트가 담당하고 있음
- 프롬프트끼리 결합도 할 수 있고 저장할 수도 있음 
- ChatPromptTemplate 클래스를 이용하여 프롬프트를 좀 더 유연하게 할 수 있고, 데이터 검증도 가능함 
```python
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate

template = PromptTemplate.from_template(
    "What is the distance between {country_a} and {country_b}"
)

prompt = template.format(country_a="Mexico", country_b="Thailand")

chat = ChatOllama(model="llama3:latest", temperature=0.1)

chat.predict(prompt)


template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a geography expert. And you only reply in {language}."),
        ("ai", "Ciao, mi chiamo {name}!"),
        (
            "human",
            "What is the distance between {country_a} and {country_b}. Also, What is your name?",
        ),
    ]   
)

template.format_messages(
    language="Greek",
    name="Socrates",
    country_a="Mexico",
    country_b="Thailand",
)
```

```
result : 

[SystemMessage(content='You are a geography expert. And you only reply in Greek.'),
 AIMessage(content='Ciao, mi chiamo Socrates!'),
 HumanMessage(content='What is the distance between Mexico and Thailand. Also, What is your name?')]
```

## 3.3 OutputParser and LCEL(LangChain Expression Language, 표현언어)
- LCEL을 사용하면 심플하게 코드 작성 가능 
- 다양한 템플릿과 LLM호출 그리고 서로 다른 응답(response)를 함께 사용하게 해줌
- OutputParser가 필요한 이유 : LLM의 응답을 변형해야할 때가 있기 떄문임 (응답을 List로 변환 가능) 또한 text로만 나오는 것을 dictionary나 tuple에 저장할 수 있게 할 수 있음 
- chain 기법을 통해 | 로 template과 chat(LLM), outputParser를 연결하여 사용

```python
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a list generating machine. Everything you are asked will be\
            answered with a comma separated list of max {max_items} in lowercase. Do NOT reply with anything else.",
        ),
        ("human", "{question}"),
    ]
)

# 원본 소스 
# prompt = template.format_messages(max_items=10, question="What are the colors?")
# result = chat.predict_messages(prompt)
# p = CommaOutputParser()
# p.parse(result.content)

# chain을 통해 코드 단축 가능 
chain = template | chat | CommaOutputParser()

chain.invoke({
    "max_items":5,
    "question":"What are the pokemons?"
})


# chain을 여러개 둘 수 있음 
chain_one = template | chat | CommaOutputParser()
chain_two = template2 | chat | outputparser 

all = chain_one | chain_two | output

chain.invoke({
    "max_items":5,
    "question":"What are the pokemons?"
})
```
```
result : 

['bulbasaur', 'charmander', 'squirtle', 'pikachu', 'charizard']
```


## 3.4 Chaining Chains
- 원리 : chain.invoke의 값이 template로 들어가고, 그 response는 chat으로 감. chat의 response는 OutputParser()로 들어가고 결과를 출력함 
- Streaming 옵션을 추가하면 결과 텍스트를 스트리밍형식으로 볼 수 있음 (StreamingStdOutCallbackHandler추가 필요)

The input type and output type varies by component:

|Component|Input Type|Output Type|
|---------|----------|-----------|
|Prompt|Dictionary|PromptValue|
|ChatModel|Single string, list of chat messages or a PromptValue|ChatMessage|
|LLM|Single string, list of chat messages or a PromptValue|String|
|OutputParser|The output of an LLM or ChatModel|Depends on the parser|
|Retriever|Single string|List of Documents|
|Tool|Single string or dictionary, depending on the tool|Depends on the tool|

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

chef_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "",
    ),
    ("human", "I want to cook {cuisine} food."),
    
])

chef_chain = chef_prompt | chat 

veg_chef_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a vegetarian chef specialized on making traditional recipies vegetarian.\
        You find alternative ingredients and explain their preparation.\
        You don't radically modify the recipe.\
        If there is no alternative for a food just say you don't know how to replace it.",
    ),
    ("human", "{recipe}"),    
])

veg_chain = veg_chef_prompt | chat 

final_chain = {"recipe": chef_chain} | veg_chain 

final_chain.invoke({
    "cuisine":"indian"
})
```

```
That sounds great! Indian cuisine is diverse and flavorful. Here are a few popular Indian dishes you might consider cooking, along with brief descriptions and basic recipes:

### 1. **Butter Chicken (Murgh Makhani)**
A rich and creamy chicken dish cooked in a spiced tomato sauce.

**Ingredients:**
- 500g chicken (boneless, cut into pieces)
- 1 cup yogurt
- 2 tablespoons butter
...
```

## 3.5 Recap
- 3장에서 배운 내용을 다시 정리 
- streaming기능 : response를 실시간으로 결과를 줌
- StreamingStdOutCallBackHandler() : 볼수 있는 문자(response)가 생길때마다 호출됨
- callback은 LLM에서 발생되는 다양한 이벤트들을 감지할 수 있음 