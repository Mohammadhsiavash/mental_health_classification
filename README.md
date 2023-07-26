# Sentiment-analysis-with-NLP
# Abstract:
In our research, we looked at how pretrained models can be used to understand mental health testimonies from a dataset called "solomonk - reddit_mental_health_posts." We wanted to see if these models rely too much on the specific labels in the data and how that affects their performance. We used five different pretrained models, named ALBERT, BERT, RoBERTa, DistilBERT, and ELECTRA, for this task. At first, the models performed well without any text filters.
However, we realized it was crucial to dig deeper into what makes the models really understand the content. So, we ran experiments where we removed parts of the text, specifically the subreddit titles that are in the text, during the evaluation of the models. When we did this, we noticed that the models' accuracy decreased. This suggests that the models depend on having direct references to the labels for correct classification. This label dependency can affect how well the models work in different situations and can also lead to bias.(difference is about 5 percent)
Interestingly, we found that removing label references didn't really change the models' decisions based on text length or the number of characters. This means that the models focus more on the meaning of the text rather than its structure.
Our research also showed that pretrained models can understand subtle details related to mental health conditions, which shows promise in using pre-trained models to find important patterns in mental health testimonies.
By understanding these aspects better, we can develop AI systems that help accurately and fairly identify mental health concerns, supporting both mental health professionals and those who need care and help. This knowledge contributes to better understanding interpretable AI and mental health informatics, and is possible to define patterns on each condition.
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


