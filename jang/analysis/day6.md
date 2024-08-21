# 10.1 AsyncChromiumLoader

- sitemap.xml에서 page url을 검색
- AsyncChromiumLoader는 url의 리스트를 받음
- loader결과 html결과나오는데 이거 그냥하면 돈많이드니까 transformer사용
- AsyncChromiumLoader+transformer는 웹사이트 스크랩할떄 유용 - 페이지 렌더링/data 가져오거나/ JS코드 처리등
  - playwrite에 headless=true하면 브라우져를 내 PC 프로세스로 수행, 느림

# 10.3 Parsing Function

- 스크랩해온 url 필터링
- beautiful soup는 똑똑한 html로 검색,삭제등이 가능해서 이거사용해서 필터링
- 필터링후 개행/공백 등을 파싱해서 리턴가능

# 10.4 Map Re Rank Chain

- 첫번쨰 chain에 필요한건 retriever에 의해 반환된 다수의 doc+ question
- getAnswer의 return이 choose_answer의 input
