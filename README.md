# Sentiment-analysis-with-NLP
# Abstract:
We looked at how pretrained models (PM) can be used to understand mental health testimonies from a dataset called "solomonk - reddit_mental_health_posts." We wanted to see if these models rely too much on the specific labels in the data and how that affects their performance. Five different PMs were used, named ALBERT, BERT, RoBERTa, DistilBERT, and ELECTRA.
We experimented how affected removing specifically the subreddit titles during the evaluation of the models, noticing that the models' accuracy decreased about 5 percent. This label-mention dependency can affect how well the models work in different situations and also lead to bias.

The models focus more on the meaning of the text rather than how much extra information we could train, therefore PM can understand subtle details related to mental health conditions, which shows promise in using PM to find important patterns in mental health testimonies.
By understanding these aspects better, we can develop AI systems that help accurately and fairly identify mental health concerns, supporting both mental health professionals and those in need.
# Team Members
Daniela Baño

Mohammad Hasan Siavash

Saloni Das

Fredy Orozco

Fernanda Zetter

# 🚀 Quick Tour

In this project we work on a sentiment analysis model with pytorch

our dataset will be : https://huggingface.co/datasets/solomonk/reddit_mental_health_posts
the sentiment analysis is on the body column of this dataset to classify them into five unique labels(ocd, ptsd, depression, aspergers and ADHD) that are in the subreddit column of this dataset.

fine-tuned ALBERT, BERT, RoBERTa, DistilBERT, and ELECTRA.

