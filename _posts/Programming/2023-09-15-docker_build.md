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







$P(x_t) = N(\mu, \sigma^2) = exp(-\cfrac{(x_t-\mu)^2}{2\sigma^2}) + C$

$log(P(x_t)) = -\cfrac{(x-m)^2}{2\sigma^2}$

$\triangledown_{x_t} log(P(x_t)) =  \triangledown_{x_t}(-\cfrac{(x_t-\mu)^2}{2\sigma^2}) = \cfrac{x_t-\mu}{\sigma^2} = -\cfrac{\sigma\epsilon}{\sigma^2} = -\cfrac{\epsilon}{\sigma}$ ($ x_t = \mu + \sigma\epsilon$)

$x_t = \sqrt{\bar\alpha_{t}}x_0 + \sqrt{1-\bar\alpha_{t}}\epsilon_{\theta}(x_t)$ ($\bar\alpha_{t} = \prod_{i=1}^{T}\alpha_i$)



$\triangledown_{x_t} log(P(x_t\|y)) = \triangledown_{x_t} log(P(x)) + \triangledown_{x_t} log(P(y\|x_t)) - \triangledown_{x_t} log(P(y))$

$\triangledown_{x_t} log(P(x_t\|y)) := \triangledown_{x_t} log(P(x)) + \gamma\triangledown_{x_t} log(P(y\|x_t))$ ($\gamma$는 scaling의 정도)

$\triangledown_{x_t} log(P(y\|x_t)) = \triangledown_{x_t} log(P(x_t\|y)) + \triangledown_{x_t} log(P(y)) - \triangledown_{x_t} log(P(x_t))$<br/>&emsp;&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$:= \triangledown_{x_t} log(P(x_t\|y)) - \triangledown_{x_t} log(P(x_t))$

$\triangledown_{x_t} log(P(x_t\|y)) = \gamma\triangledown_{x_t} log(P(x_t\|y)) + (1 - \gamma)\triangledown_{x_t} log(P(x_t))$
