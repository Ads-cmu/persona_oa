import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time 
from openai import OpenAI
import torch 

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer
 
def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   return {"accuracy": accuracy}

# Load the dataset
dataset = datasets.load_dataset("scott-persona/emotion_test_set", split="train")
text_column = "situation"
label_column = "emotion"

label2id = {label: i for i, label in enumerate(set(dataset[label_column]))}
id2label = {i: label for label, i in label2id.items()}
dataset = dataset.map(lambda example: {'labels': label2id[example[label_column]]})

emotions = set()
for _, example in enumerate(dataset):
  emotions.add(example[label_column])

train_test_split = dataset.train_test_split(test_size=0.05, seed=42)
train_set = train_test_split["train"]
test_set = train_test_split["test"]

# Initialize BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
   return tokenizer(examples[text_column], truncation=True, padding=True, max_length=512)
 
tokenized_train = train_set.map(preprocess_function, batched=True)
tokenized_test = test_set.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=27)
model.config.id2label = id2label
model.config.label2id = label2id

repo_name = "Advaith28/persona_alternate"
 
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=50,
   weight_decay=0.01,
   save_strategy="no",
   push_to_hub=False,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.evaluate())

def predict_emotion(text):
    # Preprocess the text
    inputs = tokenizer(text,truncation=True, padding=True, max_length=512)
    breakpoint()
    # Make predictions
    outputs = trainer.predict(inputs)
    print(f"Outputs shape: {outputs.predictions.shape}")

    # Convert logits to probabilities
    probabilities = torch.softmax(outputs.predictions, dim=-1)
    print(f"Probabilities shape: {probabilities.shape}")

    # Get the predicted label
    predicted_label_id = torch.argmax(probabilities, dim=-1).item()
    print(f"Predicted label ID: {predicted_label_id}")

    predicted_label = id2label[predicted_label_id]
    print(f"Predicted label: {predicted_label}")

    return predicted_label


client = OpenAI()

def get_embeddings(text):
    response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

print("Generating embeddings...")
emotion_map = dict()
for emotion in emotions:
    emotion_map[emotion] = get_embeddings(emotion) #caching values
print("Embeddings generated")


print("Performing model inference...")
train_predictions,label_ids,_ = trainer.predict(tokenized_train)
print("Model inference done.")

def cosine_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

print("Calculating accuracy...")
# Calculate the accuracy
total = 0
for i,row in enumerate(train_predictions):
    pred_emotion = id2label[np.argmax(row)]
    actual_emotion = id2label[label_ids[i]]
    similarity = cosine_angle(emotion_map[actual_emotion],emotion_map[pred_emotion])
    total+=similarity

accuracy = total / len(train_set)
print(f"Training accuracy: {accuracy * 100:.2f}%")

test_predictions,label_ids,_ = trainer.predict(tokenized_test)
# Calculate the accuracy
total = 0
for i,row in enumerate(test_predictions):
    pred_emotion = id2label[np.argmax(row)]
    actual_emotion = id2label[label_ids[i]]
    similarity = cosine_angle(emotion_map[actual_emotion],emotion_map[pred_emotion])
    total+=similarity

accuracy = total / len(test_set)
print(f"Test accuracy: {accuracy * 100:.2f}%")