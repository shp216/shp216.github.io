---
layout: single
title: "FastAPI Basic"
categories: FastAPI
tag: [FastAPI]
toc: true
author_profile: false
Typora-root-url: ../
use_math: true






---

# FastAPI Basic

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

![0901-fastapi-1](/../images/2023-09-01-fastapi_practice1/0901-fastapi-1.png)

http://127.0.0.1:8000에 들어가보면 서버가 열려서 health_check_handler함수가 실행된점을 볼 수 있고, http://127.0.0.1:8000/docs에 들어가면 FastAPI에서 자동으로 SwaggerUI를 띄어준 것을 알 수 있다.

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
```



## GET API - 전체 조회

* EX 01)

```python
@app.get("/todos")
def get_todos_handler():
    return list(todo_data.values())
```

<u>[Result]</u> - 원하는 todo_data의 value값을 리스트에 넣어서 response받은 것을 알 수 있음

![0901-fastapi-2](/../images/2023-09-01-fastapi_practice1/0901-fastapi-2.png)

### Query Parameter

응답 요청 시 query parameter를 통해 원하는 형태 혹은 원하는 값으로 받고자 할 때 이용한다

* EX 02)

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

![fastapi-4](/../images/2023-09-01-fastapi_practice1/fastapi-4.png)

* EX 03)

```python
@app.get("/todos/{todo_id}")
def get_todo_handler(todo_id: int):
    return todo_data.get(todo_id, {}) # GET할 data가 있으면 해당 data를 반환, 아니면 {}를 반환
```

![EX03-1](/../images/2023-09-01-fastapi_practice1/EX03-1.png)

![EX03-2](/../images/2023-09-01-fastapi_practice1/EX03-2.png)

## POST API - 생성

post를 이용해서 todo를 생성하는 과정이다. todo를 생성하기 위해서는 User로부터 data를 전달받아야 하는데, 이 때 필요한 것이 RequestBody이다. FastAPI에서는 pydantic의 BaseModel을 이용해서 쉽게 RequestBody를 처리할 수 있다. 

* Ex 01)

```python
from pydantic import BaseModel #BaseModel import하기

#RequestBody
class CreateTodoRequest(BaseModel): #RequestBody의 형태를 명시해준다
    id: int
    contents: str
    is_done: bool


@app.post("/todos")
#인자로 request를 명시해주면, FastAPI가 자동적으로 RequestBody를 CreateTodoRequest의 넣어서 처리!
#ResponseBody
def create_todo_handler(request: CreateTodoRequest): 
    todo_data[request.id] = request.dict() # todo_data의 type이 dict이기 때문에 맞춰줌
    return todo_data[request.id]

```

![PostAPI-01](/../images/2023-09-01-fastapi_practice1/PostAPI-01.png)

![PostAPI-02](/../images/2023-09-01-fastapi_practice1/PostAPI-02.png)

## PATCH API - 수정

기존에 있던 todo를 수정하는  API

```python
from fastapi import Body

@app.patch("/todos/{todo_id}")
def update_todo_handler(
        todo_id: int,
        is_done: bool = Body(..., embed=True) #특정 parameter만 request할경우
):
    todo = todo_data.get(todo_id)
    if todo:
        todo["is_done"] = is_done
        return todo
    return {}
```

![Patch-03](/../images/2023-09-01-fastapi_practice1/Patch-03.png)

![patch-02](/../images/2023-09-01-fastapi_practice1/patch-02.png)

* RequestBody를 설정하는 법

  1. 함수 내부에 인자를 통해 설정 -> 일부만 Request 하기 위해서는 Body를 이용

  2. class를 미리 설정해 인자에 (request: class이름) 형태로 설정하면 자동으로 RequestBody 설정

     

## Delete - 삭제

기존에 있던 todo data를 삭제하는 API

```python
@app.delete("/todos/{todo_id}")
def delete_todo_handler(todo_id: int):
    todo_data.pop(todo_id, None)
    return todo_data
```

![Delete-01](/../images/2023-09-01-fastapi_practice1/Delete-01-3733738.png)

![Delete-02](/../images/2023-09-01-fastapi_practice1/Delete-02.png)

## FastAPI Status Code

![status code](/../images/2023-09-01-fastapi_practice1/status code.png)

```python
from fastapi import HTTPException #Error처리를 위해 HTTPException import하기

@app.get("/todos/{todo_id}", status_code=200) #status_code를 명시해서 성공 시 출력 값 설정
def get_todo_handler(todo_id: int):
    todo = todo_data.get(todo_id)
    if todo:
        return todo
    #실패시 status_code 반환 값 설정 및 detail 문장을 return
    raise HTTPException(status_code=404, detail = "Todo Not Found")
    
```

