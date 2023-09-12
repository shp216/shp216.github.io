---
layout: single
title: "FastAPI_DataBase"
categories: FastAPI
tag: [FastAPI]
toc: true
author_profile: false
Typora-root-url: ../
use_math: true







---



# FastAPI Test code

<img src="/../../../../Desktop/pytest.png" alt="testcode" style="zoom:50%;" />

<img src="/../images/2023-09-09-fastapi_practice3/pytest.png" alt="pytest" style="zoom:50%;" />

## 세팅

1. pytest 설치

   ```terminal
   pip install pytest
   ```

2. src파일 안에 python package 추가

   pytest가 파일을 찾아서 test를 돌리기 위해서는 스스로 경로를 탐색해야 한다. 이 때, python package를 만들어 __init.py__ 가 필요한데 python package를 만들면 __init.py__ 가 생성되서 이를 통해 pytest가 이게 test파일이구나 라고 인식을 한다. 따라서 일판 file이 아닌 python package를 추가해준다.

   <img src="/../images/2023-09-09-fastapi_practice3/python_package.png" alt="python_package" style="zoom:50%;" />

3. Practice

   terminal에 pytest라는 명령어를 입력해서 동작시키면, pytest가 src 디렉토리에서 test코드들을 쭉 검색을 한다.  init.py가 있어야 pytest가 인식을 할 수 있으며, test를 진행하고자 하는 파일명은 test_로 시작해야 한다. 

   기존 main.py에 health_check_handler라고 GET API가 잘 동작하는지 실습했던 부분이 있다.

   ```python
   @app.get("/")
   def health_check_handler():
       return {"ping": "pong"}
   ```

   [ Test Code ]

   ```python
   from fastapi.testclient import TestClient
   
   from main import app
   
   
   client = TestClient(app=app) #우리의 app이 TestClient에 의해 test될것!
   
   
   def test_health_check():
       response = client.get("/")
       assert response.status_code == 200
       assert response.json() == {"ping": "pong"}
   ```

   ```terminal
   pytest

​		

## TEST CODE GET API 전체조회

```python
def test_get_todos():
    response = client.get("/todos")
    assert response.status_code == 200

    # ASC
    assert response.json() == {
        "shp": [
            {
                "id": 3,
                "contents": "FastAPI Section 2",
                "is_done": False
            },
            {
                "id": 5,
                "contents": "string",
                "is_done": True
            },
            {
                "id": 6,
                "contents": "string",
                "is_done": False
            }
        ]
    }

    # DESC
    response = client.get("/todos?order=DESC")
    assert response.status_code == 200
    assert response.json() == {
        "shp": [
            {
                "id": 6,
                "contents": "string",
                "is_done": False
            },
            {
                "id": 5,
                "contents": "string",
                "is_done": True
            },
            {
                "id": 3,
                "contents": "FastAPI Section 2",
                "is_done": False
            },
        ]
    }
```

## PyTest Mocking

Test를 할때마다 계속 DB와 통신을 하며 기다려야 하는 상황이 발생하기에 pytest-mock을 사용해서 가짜 데이터인 "mock"을 사용하는 것이다. 



​		

