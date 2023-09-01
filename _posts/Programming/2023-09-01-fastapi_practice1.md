---
layout: single
title: "FastAPI 실습1"
categories: FastAPI
tag: [FastAPI]
toc: true
author_profile: false
Typora-root-url: ../
use_math: true






---

# FastAPI 실습1

## 세팅

1. 가상환경 세팅

   ```terminal
   python3.10 -m venv todos
   cd todos
   source bin/activate
   ```

2. pycharm에서 해당 파일로 프로젝트 열기

   src 디렉토리를 하나 만들어주는데 보통 src파일에는 핵심code들이 내재되어있다. ex) main.py

   src 디렉토리에서 mark directory as로 들어가 Sources Root를 체크 -> 아이콘이 파란색으로 바뀜 : 파이썬 파일들을 import하는 과정에서 src파일을 root path로 잡는다.

3. fastapi, uvicorn 설치

   ```terminal
   pip install fastapi
   pip install uvicorn
   ```

   

## FastAPI 프로젝트 세팅

```python
from fastapi import FastAPI
app = FastAPI() #app에 FastAPI를 연결해서 서버를 띄움 -> 서버로 HTTP요청 가능!
@app.get("/") #FastAPI 서버에게 root path에 GET요청하는 api
def health_check_handler(): #GET요청을 하면 이 함수 실행
    return {"ping": "pong"}
```

```terminal
cd src
uvicorn main:app #main.py의 app객체를 이용할거다!
```

<u>[Result]</u> - 8000번 포트로 서버가 잘 열린 것을 확인할 수 있음

![0901-fastapi-1](/images/2023-09-01-fastapi_practice1/0901-fastapi-1.png)

http://127.0.0.1:8000에 들어가보면 서버가 열려서 health_check_handler함수가 실행된점을 볼 수 있고, http://127.0.0.1:8000/docs에 들어가면 FastAPI에서 자동으로 SwaggerUI를 띄어준 것을 알 수 있다.

## GET API - 전체 조회

```python
todo_data = {
    1: {
        "id": 1,
        "contents": "FastAPI_1",
        "is_done": True
    },
    2: {
        "id": 2,
        "contents": "FastAPI_2",
        "is_done": False
    },
    3: {
        "id": 3,
        "contents": "FastAPI_3",
        "is_done": False
    },
}


@app.get("/todos")
def get_todos_handler():
    return list(todo_data.values())
```

<u>[Result]</u> - 원하는 todo_data의 value값을 리스트에 넣어서 response받은 것을 알 수 있음

![0901-fastapi-2](/../images/2023-09-01-fastapi_practice1/0901-fastapi-2.png)

### Query Parameter

응답 요청 시 query parameter를 통해 원하는 형태 혹은 원하는 값으로 받고자 할 때 이용한다

```python
@app.get("/todos")
def get_todos_handler(order: str):
    ret = list(todo_data.values())
    if order == "DESC":
        return ret[::-1]
    return ret
```

<u>[Result]</u> - order 값에 DESC를 넣어주면 설계한 function대로 거꾸로 정렬되어서 응답을 받는 것을 알 수 있다.

![0901-fastapi-3](/../images/2023-09-01-fastapi_practice1/0901-fastapi-3.png)

![0901-fastapi-4](../images/2023-09-01-fastapi_practice1/0901-fastapi-4.png)
