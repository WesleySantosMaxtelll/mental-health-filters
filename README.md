# mental-health-filters

This repository contains the implementations of machine learning models designed to detect social media users diagnosed with depression and anxiety. The goal of this project is to provide an automated tool for identifying individuals who may be at risk based on their social media activity and language patterns. By leveraging natural language processing (NLP) techniques and machine learning algorithms, this project aims to contribute to mental health awareness and early detection.

The models can be in models.py as the following references:
- Bert = BertUserTwitter
- GPT.H = MoE with filter of high relevance
- GPT.L = MoE with filter of low relevance
- GPT.HL = MoE with filter of high and low relevance

Files trainer, combiner and tester are used to train each individual model, combiner build the combination of models with gating network, and tester runs on test samples. In order to use GPT.HL with SMHD data, please update the encoding model as well as the labels associated with each post.
