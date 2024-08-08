langchain 쓰는 이유
gpt만 쓰는것보다 데이터 전처리 등에서의 공수 절감됨

ChatOpenAI() - parameter
- temperature 가 높을수록 창의적이나 부정확한 답변
- api key
- base url

여러 gpt(openAI) model별 차이점
- 추가 공부 필요한 내용
- gpt turbo 3.5
- instructor (어투?)

chatOpenAI()
- predict : 질문을 입력하는 method
- AI message : AI 이름이나 역할 부여
- System Message : 규칙을 정해주는 메시지

Prompt를 잘 입력할수록 ai의 답변 수준이 높아짐
prompt를 잘 짤수록 ai의 성능이 좋아 prompt engineering 이라는 말도 있음
- prompt template
- chat template

output parser
- dict 형태로 나온 raw data를 파싱하여 가독성있는 답변 도출

chain : template + llmModel + outputParser
chain 끼리 결합 가능