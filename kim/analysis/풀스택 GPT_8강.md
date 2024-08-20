
# 8 private GPT
## 8.0 Introduction
## 8.1 HuggingFaceHub
hugging face 접속 > Text Generation > mistralai/Mistral-7B-v0.1 (강사가 private GPT로 추천, https://huggingface.co/mistralai/Mistral-7B-v0.1)
> 해당모델로 배포하고 싶다면 상단의 deploy 클릭
 
hugging face API에 엑세스 하려면 token이 필요함.
https://huggingface.co/settings/tokens
> 위 주소로 이동
 
 
max_new_tokens 파라미터 확인 : deploy > Interface API > Get started with Interface API > Detailed parameters> Text Generation task
 
* 모델은 반드시 Inference API 가 활성화 되어 있어야 함
 
Mistral AI 문서 보기 : Mistral-7B-v0.1 페이지에서 release blog post 클릭(https://mistral.ai/news/announcing-mistral-7b/) > developers > Docs > 
 
mistralai/Mistral-7B-Instruct-v0.1 : instruction을 따르는 모델
 
 
## 8.2 HuggingFacePipeline
8.1에서는 Hugging face로 데이터가 전송되는 방식으로 이번장에서 privateGPT 실습
모델을 다운받고 컴퓨터에서 실행시켜보자
다운받기 위한 작은모델을 선택 : gpt2(https://huggingface.co/openai-community/gpt2), 파라미터가 몇 밀리언에 불과
 
기가바이트 GPU 가지고 있다면 강력한 모델인 falcon-7b-instruct 를 사용해보는 것을 권장
 
 
##8.3 GPT4All
GPT4All은 내 컴퓨터에서 로컬 모델을 실행하는 또 다른 방법임
 
 
## 8.4 Ollama
llm을 로컬로 사용하는 강사가 가장 선호하는 방식
ollama 덕분에 documentGPT를 privateGPT로 바꾸는데 단 세줄의 코드만 필요
> ollama.ai로 이동 > ollama download 
 
> models로 이동 > mistral(강의에 사용)
 
falcon 180 billion parameter 모델은 chat GPT4의 성능을 내는 오픈소스모델!
실행하기에는 용량이 매우 큼(192GB), 실행명령어는 ollama 설치 후 ollama run falcon을 입력
 
mistral은 최소 8GB의 메모리가 필요
 
Ollama가 작동하는 방식은 내 컴퓨터에 서버를 만드는 거임, langchain은 ollama를 위한 wrapper를 가지고 있음
 
langchain이 curl, post, request등을 모두 처리 = 강력하다!

터미널 > ollama run llama3.1, ollama run mistral

(GHAT GPT)mac 에서 ollama run 으로 모델 설치하면 저장돼?

ChatGPT said:
ChatGPT
macOS에서 ollama run 명령어를 사용하여 모델을 설치하면, 설치된 모델은 기본적으로 다음 경로에 저장됩니다: ~/Library 폴더는 일반적으로 숨겨져 있으므로, Finder에서 "이동" 메뉴를 사용하여 "폴더로 이동"을 선택하고 해당 경로를 입력하면 접근할 수 있습니다.
 : ~/Library/Application Support/Ollama/models

/bye : ollama run 종료 > 터미널에 ollama입력 : ollama 제공 많은 명령어 볼 수 있음
Ex) ollama ls : 내 컴퓨터에 있는 모델과 크기 볼 수있음
Ollama rm : 모델 삭제(ollama rm llama3.1:latest)

ChatPromptTemplate.from_messages : 채팅모델이라기 보다 그저 instructions를 따르는 instruce에 가깝다.
    > ChatPromptTemplate.from_messages로 변경 : 이건 그저 string을 취급

 
LangChain과 Ollama 덕분에 두줄의 코드만으로 private GPT 구현


# 8.5 Conclusions

Ollama를 사용하는 것이 괜찮거나 좋다면 다시 chatGPT나 GPT3, GPT4를 사용하러 갈 필요는 없다.
Answer 들이 가끔 펑키할 때도 있는데 그때는 prompt를 좀 수정해야 한다.





그룹스터디
unslot : 파인튜닝 빠르게 할 수 있도록
