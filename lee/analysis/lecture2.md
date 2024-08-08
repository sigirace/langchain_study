# 1. Introdution


## (1) Requirements
- 이미 만들어진 Model들을 활용할 것임

### **Langchain**
> LLM(Large Language Model) 을 활용한 application 제작에 사용되는 Framework 

- LangChain에는 메모리를 위한 모듈이 별도 존재하나, GPT-4에는 없어 직접 구현해야 함.   
- Document를 위한 모듈과 Embedding, Chat model을 위한 모듈이 존재   
- LLM application 의 모든 필수요소들 사이의 호환을 위한 중간 계층 역할을 수행

### **Streamlit**
> python 코드 작성만으로 UI 제작이 가능

- html, css, javascript 불필요

### **Pinecone**
> Vector Database (무료 버전 사용)

### **Hugging Face**
> GPT4 외 다른 모델을 가져오는 용도로 학습

### **FastAPI**
> ChatGPT plugin API 구축을 위해 사용 (GPT application을 위함)


## (2) Virtual Environment
작업 중인 폴더에 반드시 **.gitignore**를 만들어 줄 것   
그리고 내용에 `/env`를 넣어 env 디렉토리가 gitignore 대상이 되도록 함

1. 작업 중인 폴더에 `git init` 명령어 입력
- 가상환경은 Package들의 설치를 독립시키기 위해 필요함    
- (+) PC 내 설치된 하나의 엔진/FW만을 사용하는 게 아닌, 프로젝트 별 독립적인 개발환경 구축을 위함
2. 가상환경 구축을 위해 `python -m venv ./env` 입력
- (+) python이 설치된 가상환경 'env'를 현재 폴더에 구축함
3. requirements.txt 
- 프로젝트의 dependency package를 명시한 텍스트 문서 (강의 내 제공)
- 명시된 package들의 설치를 위해 `pip install -r requirements.txt` 입력
- gitignore에 포함하지 않음

# Trouble Shooting
- venv 구축 간 python 3.12 버전 이슈로, conda를 설치해 python3.11환경을 따로 만들어 주었음. 해당 가상환경에서 venv 명령어를 통해 env 생성하였음.

- `pip install -r requirements.txt` 중 아래의 dependency 설치 간 오류 발생
    - cffi==1.15 : 1.16
    - numpy==1.25.2 : latest (1.26.4)
    - onnx==1.15.0 : latest
    - onnxruntime==1.16.0 : latest (1.18.1)
    - pandas==2.1.0 : latest (2.2.2)
    - pulsar-client==3.3.0 : latest (3.5.0)
    - rpds-py==0.16.0 : : latest (0.19)
    - sentencepiece==0.1.99 : latest (0.2)