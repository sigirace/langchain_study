# 3.1 OpenAI 연결

- openAI API를 사용하면 memory등의 이점이 없음, 코드작성자가 다 처리해야함
- langchain을 사용하면 다양한 세팅들을 알아서 처리해줌

# 3.2 langchain prompt 사용법

- prompt는 llm과의 소통창구로 prompt 성능에 따라 llm의 답변 수준이 정해짐
- promtTemplate = system + ai + human
  - system message : llm의 행동 제어( 어떻게 행동해야하는지에 대한 지시사항)
  - ai message : 사용자 메세지
  - human message : 이전에 어떤 대답을 했는지를 기록하는 데 사용, 대화의 맥락을 이해하고 더욱 적절한 응답

# 3.3 Output Parser

- Output Parser의 사용이유 : LLM response 변형용(llm은 보통 text로 답함)
- answered with a comma separated list등을 system message에 추가해서 답변형태를 강제 할 수 있음
- chain = template+llmModel+outputParser
- LCEL(LangChain Expression Language)
  - Langchain에서 제공하는 기능들을 조합한 Chain을 마치 블록처럼 쉽게 분해, 조립할 수 있도록 설계한 프레임워크
  - Langchain계의 scikit-learn

# 3.4

- chain = prompt+retriever+llm+tool+parser
- final_chain=chef_chain|veg_chain
- chef에서 얻은 recipe(output)이 veg의 input으로사용
- 이 final_chain을 쓰면 invoke를 두번해야함
- 아래녀석도 코드상으로만 1번호출이지 실제 2번호출함
