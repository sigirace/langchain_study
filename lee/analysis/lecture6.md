# 06. RAG_DocumentGPT

## RAG (Retrieval Augmented Generation, 검색 증강 생성)

- model들은 많은 데이터를 통해 학습하지만 DB나 문서와 같은 private data에 접근할 수는 없다.
- model은 이러한 data를 알 수 없으므로, 우리는 질문할 때 그와 관련있는 문서들도 같이 준비하게 된다. (vector나 저장기 활용은 뒤에서 배운다.)

### What is 'foo'?

- 우리의 model은 'foo'라는 단어를 아예 모르는 상태이다.
- 따라서 우리는 'foo'와 관련된 문서들을 준비해 가져올 것이다.
- **이 문서들을 context로써 질문과 함께 묶어 prompt를 구성해, Language Model에 보낼 것이다.**
- Model은 기존에 학습된 data와 추가로 전달 받은 context로써의 data를 모두 갖게 된다.
- 이러한 개념을 ***RAG*** 이라고 한다.

## Retrieval
- Langchain의 모듈
- Source -> Load -> Transform -> Embed -> Store -> Retrieve
  
### UnstructedFileLoader
- 대부분 파일들에 호환되는 범용 loader

### 문서 분할

- 예제의 경우 전체 챕터가 하나의 문서에 들어가 있다. 
- 너무 큰 한 덩어리 데이터를 분할하여 파일의 필요한 부분만을 사용하고자 분할한다.
- `chunk_size`를 지정해 잘라낸 각 덩어리는 작아졌지만 문단까지 잘려 그냥 사용하기엔 문제가 많았음.
- `chunk_overlap` 을 지정해 앞 chunk의 일부를 뒤 chunk의 시작으로 끌고올 수 있다. 단, 이로 인해 chunk 간 중복되는 부분이 생긴다.

### Token

위에서 text를 chunk 단위로 분류했지만, 엄밀히 token과 letter는 다른 것이다.
- token은 여러 개의 문자나 단어가 될 수도 있다.

문장을 Token으로 분리했을 때, Model이 받게 되는 데이터는 각 Token에 대한 ID값이다.
- nomad : nom / ad -> [17101, 329] 와 같은 식
- 이를 기준으로 AI는 'token_17101 다음에 token_329가 올 확률이 높다' 등의 예측 가능

### Tiktoken

이러한 데이터의 Tokenizing을 위해 사용하는 라이브러리

## Embedding

지금까지 위의 RAG 중 Load-Transform 단계까지는 수행했다.   
이제 컴퓨터가 이해할 수 있도록 data를 Embedding해 주어야 한다.

문서의 split 처리된 data마다 각각의 (3D)vector를 만들 것이며, 이를 위해 embedding model을 사용한다.

Masculinity | Femininity | Royalty 를 우리의 3개 차원으로 둔다고 가정해 보자.
여러 단어들을 아래와 같이 우리의 차원에서 정의해 보자.

[word(Masculinity, Femininity, Royalty)]
- King  (0.9, 0.1, 1.0)
- Queen (0.1, 0.9, 1.0)
- Man   (0.9, 0.1, 0,0)

'우리의 차원에서 King은 0.9의 남성성, 1.0의 충성심, 0.1의 여성성을 가진다' 라는 것을 3차원 vector로 나타내었다.   

- Vector Calculus
King-Man = (0, 0, 1.0) = ***Royal***   
이는 'King과 Man은 우리의 차원에서 Royalty=1.0의 차이를 갖는다.' 로 해석할 수 있으며,   
우리의 차원에서 이러한 단어는 'royal'과 같이 정의내릴 수 있다.   

Man+Royal = (0.9, 0.1, 1.0) = ***King***   
이처럼 단어 간의 연관성(가깝고 멂)을 벡터 연산을 통해 나타낼 수 있다.   
(+) : 벡터 간 연관성은 벡터 간 거리로 판단한다.   

이를 통해 특정 단어에 대해 연관성이 높은 단어를 **Search**할 수 있게 된다.   
(+) 물론 연관성의 range에 대한 것은 개발자가 정해야 할 것 같다.   

많은 추천 알고리즘은 이와 같은 방식으로 동작한다.   


차원의 개수가 늘어날수록 한 단어가 담고 있는(표현할 수 있는) 속성의 개수가 많을 것이고,   
이를 통해 비슷한 영역에 있는 단어나 문서를 찾을 수 있다.   
(참고 : The conspiracy to make AI seem harder than it is! By Gustav Soderstrom (Spotify R&D) (Youtube))


## Vector Store

일종의 DB로 생각할 수 있다. Embed를 통해 vector를 만들고 이곳에 저장한 후,   
vector store에서 검색할 수 있게 된다. (관련있는 문서만 찾아낼 수 있게 된다.)

매 코드 실행마다 문서를 embed하기 보다는, embed한 내용을 저장하여 사용하자.   
Langchain에서는 embed 결과를 cache에 저장할 수 있도록 기능을 제공해 준다.

우리는 Chroma를 사용할 것이다. Local에서 실행될 것임.   

vectorstore로 보내지는 문서의 크기에 따라 비용이 달라지므로,   
의미가 왜곡되지 않는 선에서 적정한 split chunksize를 정하는 것이 중요할 것으로 보임.


## RetrievalQA

- 앞선 강의에서 사용했던 LLMChain은 Legacy에 해당함
- 따라서 Langchain에서는 LCLE Chain을 사용하는 것을 권장하고 있음.

- off-the-shelf chain으로 property를 전달하면 Document chain이 만들어져 즉시 답변됨
- 하지만 다소 모호하고, 커스터마이징이 어려움.

- System Message는 자동으로 넣어주는 것으로 확인됨.

### Document chain의 생성 방법

1. Stuff
- document로 prompt를 채우는(stuffing) 단순한 방법
- 사용자의 질문에 문서를 별도 프로세싱 없이 엎어 넘기는 방법

2. Refine
- 질문과 관련된 document를 얻어, **각각의 document를 읽으면서 질문에 대한 답변 생성을 시도**함.
- 만나는 모든 document를 통해 question을 개선시킨다.
- 즉, 질문만을 기반으로 첫 답변을 생성하고, retrieve된 document를 차례로 보며 답변을 refine해가는 방식이다.
- chain 내부에서 document의 수만큼 질의를 수행하므로 cost가 더 높음.

3. Map Reduce
- 질문에 대한 Document들을 입력받아 개별적으로 요약 작업을 수행한다.
- 각각의 요약본을 LLM에 전달한다.

4. Map Re-Rank
- 질문에 대한 Document들을 입력받아 각각의 답변을 생성하고, **각 답변에 점수를 부여**한다.
- 최종적으로 가장 높은 점수를 획득한 답변과 그 점수를 함께 반환한다.

(10:45) Chroma를 FAISS로 바꿔주었음


## LCEL Chain

- Retriever의 input은 string, output은 list:documents 이다.
- 