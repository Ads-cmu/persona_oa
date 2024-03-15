import openai
import datasets
from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
import torch
import numpy as np 

dataset = datasets.load_dataset("scott-persona/emotion_test_set",split="train")

# Load the fine-tuned model and tokenizer
model_path = "../finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, load_in_4bit=True)

def generate_and_decode(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    # Get the last generated token ID
    predicted_token_id = logits[:, 0, :].argmax(dim=-1)
    # Decode the token ID to text
    predicted_label = tokenizer.decode(predicted_token_id).strip()
    return predicted_label

openai.api_key = ""

def get_embeddings(text):
    # Generate embeddings using the "ada-2" model
    response = openai.Embedding.create(
        input=text,
        model="ada-2"
    )
    # Extract the embedding vector from the response
    embedding = response['data'][0]['embedding']
    return embedding

emotions = set()
for _, example in enumerate(dataset):
  emotions.add(example['emotion'])

emotion_map = dict()
for emotion in emotions:
    emotion_map[emotion] = get_embeddings(emotion) #caching values
    
# Apply the prediction function to the training set
dataset = dataset.map(lambda x: {"predicted_emotion": generate_and_decode(x['text']),"true_emotion":x['emotion']})

# Calculate the accuracy
total = 0
for i,row in enumerate(dataset):
    pred_emotion = row['predicted_emotion']
    if pred_emotion in emotion_map:
        embedding2 = emotion_map[pred_emotion]
    else:
        embedding2 = get_embeddings(pred_emotion)
    similarity = np.dot(emotion_map[row['emotion']],embedding2)
    total+=similarity

accuracy = total / len(dataset)
print(f"Training accuracy: {accuracy * 100:.2f}%")


