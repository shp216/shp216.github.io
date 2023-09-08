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

# FastAPI_DataBase

<img src="/../images/2023-09-04-fastapi_practice2/fastapi_db_1.png" alt="fastapi_db_1"  />

<img src="/../images/2023-09-04-fastapi_practice2/fastapi_db_2.png" alt="fastapi_db_2" style="zoom: 50%;" />

```terminal
pip install pymysql #python과 mysql을 연동
pip install cryptography #python통해 mysql에 접속할 때, 인증이나 암호등을 처리
```

## MySQL 컨테이너 실행

<u><Docker 실행 순서></u>

Dockerfile 생성 -> Dockerfile, main code, dependencies 등을 포함한 Docker Image 생성 -> Image를 이용해Docker Container 실행

```terminal
docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=todos -e MYSQL_DATABASE=todos -d -v todos:/db --name todos mysql:8.0
```

docker run: docker 컨테이너를 동작시키는 명령어

-p: port를 연결해주는 명령 (MySQL은 3306번 port를 이용한다.)<br/> local 3306 port <--> docker container 3306 port 서로를 연결

-e: 환경변수 option<br/>-e MYSQL_ROOT_PASSWORD=todos : 별도로 계정을 생성하면 상관없지만 여기서는 기존에 있는 Root계정을 사용하기에 PASSWORD 설정 해줘야함!<br/>-e MYSQL_DATABASE=todos : docker 컨테이너 실행 시, todos라는 이름의 MySQL DB를 실행해달라고 하는 것

-d: detach option으로 명령어 실행 시 container가 background에서 실행되도록 함

-v todos:/db: volume option으로 일반적으로 docker container는 container를 삭제하면 db도 삭제가 된다. 그러기에 todos라는 이름의 volume을 생성해서 local device에 data를 남기는 것

--name todos: MySQL을 동작시킬 Container의 이름

mysql:8.0 : docker image 설정

## MySQL 접속

```terminal
docker exec -it todos bash
mysql -u root -p #root user로 접속해서 port연결
Enter password: #설정한 root password 입력
-> mysql 연결 완료!

use todos;
-> todos 이용
```

이후, table을 create한 후 형식에 맞게 insert를 해주면 data가 입력되는 것을 볼 수 있다. 이 방법은 mysql에서 직접 data를 생성한 경우이고 ORM을 이용한다면 파이썬에서 class객체를 사용해서 직접 table을 관리할 수 있어서 훨씬 용이하고 편리하다. 따라서 python에서 DB table을 다루기 위해 sqlalchemy를 이용해 DB와 통신할 수 있도록 한다.

## sqlalchemy로 DB연결

sqlalchemy는 ORM을 사용할 수 있도록 도와주는 DB toolkit이다. ORM은 객체와 DB의 관계를 매핑해주는 것으로 Class를 코드에서 선언하면 DB와 매핑해서 사용할 수 있다.

1. engine을 생성해준다. -> DATABASE_URL을 잘 매핑해줘야 함!
2. sessionmaker에게 engine을 달아줘야 함! -> bind = engine
3. sessionmaker로 instance 객체를 만들어서 db와 통신한다

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

DATABASE_URL = "mysql+pymysql://root:todos@127.0.0.1:3306/todos"
engine = create_engine(DATABASE_URL, echo=True)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

```python
session = SessionFactory()
session.scalar(select(1))
```



sqlalchemy를 통해 db에 접근하려면 engine을 만들어줘야 함.<br/>query가 잘 동작하는지 어떤 query가 동작하는 지 확인하기 위해 echo=True<br/>sessionmaker는 session을 생성해서 instance를 통해 db와 통신하도록 함

## ORM 구현

ORM은 class를 DB table과 class를 연동해서 사용하는 기술이다. ORM으로 활용되는 class는 아래와 같이 구현하면 된다. Base는 무조건 상속받아야 하고, table 이름을 작성한 뒤 Column을 작성해주면 된다. Repr 함수는 python에 존재하는 representation 메소드로 꼭 작성해 줄 필요는 없으며 원하는 데이터를 예쁜 형태로 볼 수 있다. 

```python
from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy import select


Base = declarative_base()


class ToDo(Base):
    __tablename__ = "todo"

    id = Column(Integer, primary_key=True, index=True)
    contents = Column(String(256), nullable=False)
    is_done = Column(Boolean, nullable=False)

    def __repr__(self):
        return f"ToDO(id={self.id}, contents={self.contents}, is_done={self.is_done})"
```

```python
session = SessionFactory()
session.scalars(select(ToDo)) #ToDo의 모든 데이터를 조회
```



## ORM 적용해서 GET API

API안에서 session 객체를 이용해 DB에 접근하기 위해서는 Generator를 만들어줘야 한다. Generator를 만들어서 API의 parameter로써 받아준다. 이 때, Depends를 import해서 받아준다. 

Generator에서 session instance를 생성한다. 그리고 이 Generator를 API의 parameter에서 Depends를 이용하여 받아준다. 그러면 API 내에서 session을 통해 DB와 통신이 가능하다.

* main.py -> FastAPI 서버와 통신하며 API를 구현하는 곳
* repository.py -> DB에서 실행될 코드를 구현하는 곳
* connection.py -> session을 만드는 곳이라고 생각하면 편하다.<br/>1. engine 생성<br/>2. sessionmaker에 engine 달아줌<br/>3. sessionmaker의 객체 만들고 동작하도록 Generator생성
* orm.py -> DB table에 매핑되는 class 선언

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+pymysql://root:todos@127.0.0.1:3306/todos"
engine = create_engine(DATABASE_URL, echo=True)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()
```

```python
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session
from database.orm import ToDo


def get_todos(session: Session) -> List[ToDo]: #ToDo를 List에 담아서 반환한다
    return list(session.scalars(select(ToDo)))

```

```python
from fastapi import FastAPI, Body, HTTPException, Depends

@app.get("/todos", status_code=200)
def get_todos_handler(
        order: str | None = None,
        session: Session = Depends(get_db),
):
    # ret = list(todo_data.values())
    todos: List[ToDo] = get_todos(session=session)

    if order and order == "DESC":
        return todos[::-1]
    return todos
```

![todos](/../images/2023-09-04-fastapi_practice2/todos.png)

-> get_todos 함수를 통해 DB에 있는 모든 값을 list에 넣어서 모두 받아온 값은 위의 결과와 같고 코드에서는 이를 todos라고 정의하였다. 실제 현업에서 코드를 짤 때는, response를 한번 더 분리하고 정리한다고 한다. 이유는 실제는 Column이 훨씬 더 복잡할 뿐더러, Column간의 연산이 있거나 받고 싶은 값만 받고 싶다는 등 특이한 경우에 대해 유연하게 대처하기 위해 response를 분리해서 사용한다. response를 분리해서 사용할 경우에는 따로 response만 받는 response.py를 만들어 main.py에서 API내에 class객체를 사용해 값을 받으면 된다.

## 1. GET API with ORM

* Ex 01) GET API with ORM 전체 조회

```python
from pydantic import BaseModel
from typing import List


class ToDoSchema(BaseModel):
    id: int
    contents: str
    #is_done: bool

    class Config:
        orm_mode = True


class ListToDoResponse(BaseModel):
    shp: List[ToDoSchema]

```

```python
@app.get("/todos", status_code=200)
def get_todos_handler(
        order: str | None = None,
        session: Session = Depends(get_db),
):
    # ret = list(todo_data.values())
    todos: List[ToDo] = get_todos(session=session)

    if order and order == "DESC":
        return ListToDoResponse(
            shp=[ToDoSchema.model_validate(todo, from_attributes=True) for todo in todos[::-1]]
        )
    # return todos
    return ListToDoResponse(
        shp=[ToDoSchema.model_validate(todo, from_attributes=True) for todo in todos]
    )
```

![response-1](/../images/2023-09-04-fastapi_practice2/response-1.png)

-> DB에서 전체 Key들에 대해 todos에 받아온 다음, is_done을 제외한 값들을 받고 싶은 경우 이와 같이 response.py를 이용해 원하는 형태의 response를 Class로 작성하면 실제 GET API를 작동시켰을 경우, 원하는 형태로 response를 받을 수 있는 점을 확인할 수 있다. 

* Ex 02) GET API with ORM 단일 조회

```python
def get_todo_by_todo_id(session: Session, todo_id: int) -> ToDo | None:
    return session.scalar(select(ToDo).where(ToDo.id == todo_id))
```

```python
@app.get("/todos/{todo_id}", status_code=200)
def get_todo_handler(
        todo_id: int,
        session: Session = Depends(get_db)
) -> List[ToDoSchema]:
    # todo = todo_data.get(todo_id)
    todo: ToDo | None = get_todo_by_todo_id(session=session, todo_id=todo_id)
    if todo:
        return [ToDoSchema.model_validate(todo, from_attributes=True)]
    raise HTTPException(status_code=404, detail="Todo Not Found")
```

![GET_API with ORM](/../images/2023-09-04-fastapi_practice2/GET_API with ORM2.png)

## 2. POST API with ORM

실습1에서 POST API를 구성할 때 main.py안에 직접 class를 만들어 RequestBody를 만들었다. 하지만 실제로는 위의 GET API에서처럼, ResponseBody, RequestBody는 따로 Schema directory안에 파일을 따로 만들어서 보관하는게 일반적이라고 한다. 따라서 main.py안에 있던 CreateToDoRequest Class를 Refactoring해준다.

class를 드래그해서 우클릭한 후, Refactor에 들어가서 move를 클릭해서 원하는 경로값을 설정해주면 된다.

<img src="/../images/2023-09-04-fastapi_practice2/fastapi orm 정리.jpg" alt="fastapi orm 정리" style="zoom:33%;" />
