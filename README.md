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


# Explain Code and Procedure

### Load the dataset from huging face
```
dataset = load_dataset("solomonk/reddit_mental_health_posts")
```

### Pre-processing

```
def not_none(example):
    return example['body'] is not None
# At first we deleted none values in dataset

dataset = dataset.filter(not_none)

# Here we removed rows that are bigger than 500 char to make the training process faster

def filter_text(example):
  len_body = len(example['body'])
  if len_body>=500:
    return True
  return False

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def prepare_dataframe(df):
  # Concatenate title and body
  df['body'] = df.body.fillna('')
  df['body'] = df.body.str.cat(df.title, sep=' ')

  # Removed deleted posts

  df = df[~df.author.str.contains('
')]
  df = df[~df.body.str.contains('
')]
  df = df[~df.body.str.contains('
')]
  df = df[~df.body.str.contains('
')]

  # Removed moderador posts

  df = df[df.author!='AutoModerator']

  return df[['body', 'subreddit']]
```

### Split Dataset to Train and Test set

```
df = dataset["train"].to_pandas()
df = prepare_dataframe(df).reset_index(drop=True)
dataset = Dataset.from_pandas(df)

dataset_sampled = dataset.train_test_split(test_size=0.7, seed=42)['train']

train_val_test = dataset_sampled.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_test['train']
test_val_dataset = train_val_test['test']

test_val_split = test_val_dataset.train_test_split(test_size=0.5, seed=42)
validation_dataset = test_val_split['train']
test_dataset = test_val_split['test']

# Delete unnecessary columns of the dataset
columns_to_keep = ['body', 'subreddit']

columns_to_remove = [col for col in dataset_sampled.column_names if col not in columns_to_keep]

train_dataset = train_dataset.remove_columns(columns_to_remove)
validation_dataset = validation_dataset.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)
```
### Pass Data for Tokenizing
```
le = LabelEncoder()

le.fit(dataset_sampled['subreddit'])

def encode_labels(example):
    example['subreddit'] = le.transform([example['subreddit']])[0]
    return example

train_dataset = train_dataset.map(encode_labels)
validation_dataset = validation_dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)

# use pretrained tokenizer of distilbert
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def prepare_data(example):
    encoding = tokenizer.encode_plus(
        example['body'],
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt',
    )
    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(example['subreddit'], dtype=torch.long)
    }
# Map the Dataset
train_dataset = train_dataset.map(prepare_data)
validation_dataset = validation_dataset.map(prepare_data)
test_dataset = test_dataset.map(prepare_data)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
```
### Making the model
we use distilbert pretrained and fine-tune it
```
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
result = trainer.evaluate(test_dataset)
```

### computing accuracy
```
# Extract the predicted labels and true labels

# predictions = torch.argmax(torch.tensor(result.predictions), dim=1).tolist()
prediction_output = trainer.predict(test_dataset)

predictions = np.argmax(prediction_output.predictions, axis=-1)
labels = prediction_output.label_ids
losses = prediction_output.metrics
test_targets = test_dataset['labels'].numpy()

# Calculate evaluation metrics
accuracy = accuracy_score(test_targets, predictions)
f1 = f1_score(test_targets, predictions, average='macro')
# roc_auc = roc_auc_score(test_targets, result.predictions, multi_class='ovo')
recall = recall_score(test_targets, predictions, average='macro')
precision = precision_score(test_targets, predictions, average='macro')
conf_matrix = confusion_matrix(test_targets, predictions)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
# print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
```

### Ploting
```
import matplotlib.pyplot as plt
import seaborn as sns

candidate_labels = ['aspergers', 'ADHD', 'OCD', 'ptsd', 'depression']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=candidate_labels, yticklabels=candidate_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```
