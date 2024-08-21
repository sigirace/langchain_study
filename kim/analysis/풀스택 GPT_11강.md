# 11 MeetingGPT

## 11.0 Introduction
### Meeting GPT를 통해 사용자는 회의 영상이나, 사람들이 대화하는 팟캐스트를 업로드 할 수 있다.
사용자는 대화의 요약본을 받게되고 대화 중 어떤 일이 일어났는지 질문할 수 있다.

>> 사용자가 동영상 업로드 > 영상부분 제거 후 오디오만 남김 > 오디오를 10분 단위로 분할 > openAI API를 이용해 Whisper 모델(openai.com/research/whisper)에 입력 > Whisper 모델이 전체 대화를 받아적고 그 내용을 우리에게 넘겨줌 > chain을 구현하여 전체 대화 요약 > 문서를 embed > 사용자가 다른 chain을 다시 실행하도록 함(stuff, map reduce, map rerank와 같은 chain)


>> 오픈소스를 로컬에 다운받아 사용할 수도 있지만 openAI 플랫폼에 호스팅된 버전의 whisper를 이용하면 빨리 사용 가능(1분에 0.006달러)



## 11.1 Audio Extraction

FFmpeg : 컴퓨터에 다운받아 사용 가능한 CLI 도구 : ffmpeg.org > 동영상 압축, 썸네일 얻기, 오디오 추출 가능 > mp4 파일을 avi 비디오 파일로 전환도 가능 > 비디오 작업 개발자라면 필수
맥북 설치 : brew install ffmpeg : ffmpeg 명령어로 설치확인

ffmpeg -i files/podcast.mp4 -vn files/audio.mp3 : vn옵션은 영상을 무시하라는 뜻(오디오만 추출)

import subprocess #파이썬 코드에서 command 실행가능


## 11.2 Cutting The Audio
### pydub이라는 패키지를 이용해서 11.1에서 만든 오디오 파일을 10분 길이의 mp3파일들로 변환

pydub을 통해 파일을 리스트처럼 조작가능 : 오디오 파일 첫 10초, 마지막 5초 등을 추출, 파일의 볼륨도 조작가능, fade in/out 과 같은 효과도 넣을 수 있다.

4:46초까지 수강



## 11.3 Whisper Transcript
### 11.2 에서 생성한 파일청크 8개 중 두개를 openAI API로 보내서 Whisper 모델을 사용하여 녹취록 받기

6:55초까지 수강


## 11.4 Recap


## 11.5 Refine Chain Plan


agent 부분이 12강부터 나옴(중요한 내용), function call 은 명시적으로 포함여부를 판단해서 함수 호출 했지만 agent는 이런부분을 자동으로 알아서 해준다.

langsimth? : https://smith.langchain.com/
AgentExecuter


