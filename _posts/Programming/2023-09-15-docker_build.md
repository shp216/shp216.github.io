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

docker build --platform linux/amd64 -t uime .

docker image tag uime:latest shp216/uime:latest

docker push shp216/uime:latest 

-> dockerhub에 image 올라감

docker pull shp216/uime:latest

docker run -d -p 8080:8080 shp216/uime:latest