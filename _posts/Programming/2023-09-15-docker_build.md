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

경로: IME (NOT SRC)

docker-compose up --build -d

docker build --platform linux/amd64 -t (boong-fastapi) .---->얘만 해라

docker image tag boong-fastapi:latest shp216/boong-fastapi:latest

docker push shp216/boong-fastapi:latest --->얘도 해라

-> dockerhub에 image 올라감

 bdocker pull shp216/boong-fastapi:latest

docker run -d -p 8080:8080 shp216/boong-fastapi:latest

curl -X POST -H "Content-Type: application/json" -d '{"gender": "value1", "age": 123, "atmosphere" : "hello", "karlo_img" : "fqef", "up" : "123", "bottom": "123"}' http://localhost:8080/users
