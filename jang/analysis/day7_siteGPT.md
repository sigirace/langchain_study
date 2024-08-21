# 11.1 ~ 11.2

- ffmpeg를 활용하여 동영상 핸들링
- AudioSegment로 추출한 오디오 핸들링

# 11.3 Whisper Transcript

- whisper를 사용하여 텍스트 추출(JSON object형태)

# 11.5 Upload UI

- cache : 개발하는동안만 비용아끼기 위해하는거네

# 11.6 Refine Chain Plan

- 요약할때 transcript 쪼갠 두개 요약을 합치는게 아니라 1개 요약하고 요약된애에다가 다음 스크립트 업데이트하는식으로 진행

# 11.7 Refine Chain

- refine : 입력 문서를 순회하며 반복적으로 답변을 업데이트하여 응답을 구성
- 영상 업로드 > 음성 변환 > 영상 쪼개기 > transcript 생성(whisper) > 요약 > QnA
- LLM에게 첫번쨰 doc만 요약 요청(first summary chain) > doc의 숫자대로 추가 요약(refine chain)
