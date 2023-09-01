---
layout: single
title: "Transformer"
categories: PyTorch
tag: [Pytorch]
toc: true
author_profile: false
Typora-root-url: ../
use_math: true





---

# Attention is all you need

이전까지는 input data X를 가지고 output data Y를 표현하는 방식은 대표적으로 MLP, CNN, RNN등이 있었다. 각각 다른 모델이지만 결국에는 input data X에 대한 Weighted Sum으로 표현된다는 방식은 동일했다. 과연 Weighted Sum으로 표현하는 방법이 가장 좋은 output을 도출하는 방법일까? 라는 질문에서 시작된 모델이 바로 Transformer이다.

Basic Assumption: the input x can be split into multiple elements that are organically related to each other.<br/>ex)&nbsp;$\circ\$ People in Society<br/>&nbsp;&nbsp;&nbsp;&emsp;$\circ\$ Words in Sentence<br/>&nbsp;&nbsp;&nbsp;&emsp;$\circ\$ Frames in Video

그동안 우리는 예를 들어 "뛴다"라는 표현을 하나의 vector로 embedding하는 과정을 거쳤다. 하지만 사람이 어느 소속에 있느냐에 따라 역할과 행동이 다르듯이, "뛴다"라는 표현도 맥락에 따라 다르게 표현된다. 예를 들어 "운동장을 열심히 뛴다"와 "몸값이 말도 안되게 뛴다"에서 "뛴다"의 표현은 다르다. 그래서 한 사회 안에서 사람간의 관계가 있고, 문장안에 단어간의 관계가 있듯이, 각각의 Element들은 속해있는 Context안의 다른 Element들과의 관계를 고려해서 own representation을 정의한다. 이를 self-attention 기법이라고 정의한다.

## Self - Attention

Each element learns to refine its own representation by attending its context (other elements in the input).

$\circ\$ More Specifically, as a weighted sum of other elements (not arbitrary weights) in the sentence.

<img src="/images/2023-09-01-transformer/self-attention.png" alt="self-attention" style="zoom:50%;" />

Input token {$x_1, x_2, \dots, x_N$}이 존재할 때, 각각의 Input token $x_i$에 대해서 linear transformation을 통해 우리는 <u>Query, Key, Value</u>값들을 직접 mapping한다. 

Weight Paramerter W_Q, W_K, W_V는 