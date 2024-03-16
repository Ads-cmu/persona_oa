from openai import OpenAI
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch
import time 
import numpy as np

dataset = datasets.load_dataset("scott-persona/emotion_test_set",split="train")

# Load the fine-tuned model and tokenizer
model_path = "Advaith28/persona_oa_3epoch"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",torch_dtype=torch.float16, load_in_4bit=True)
model.resize_token_embeddings(len(tokenizer))
config = LoraConfig.from_pretrained(model_path)
# Load the LoRA model
inference_model = PeftModel.from_pretrained(model, model_path) 

def format_data(sample):
    instruction = "Classify the paragraph into one emotion."
    context = f"Paragraph: {sample['situation']}.\n Emotion: "
    # join all the parts together
    prompt = "".join([instruction, context])
    return prompt

# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_data(sample)}"
    return sample

def generate_and_decode(prompt):
    #print("input text: ",prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    # Get the first generated token ID
    predicted_token_id = logits[:, prompt_length-2, :].argmax(dim=-1)
    # Decode the token ID to text
    predicted_label = tokenizer.decode(predicted_token_id).strip()
    #print("model output: ",tokenizer.decode(logits.argmax(dim=-1)[0]))
    #print("predicted label: ",predicted_label)
    return predicted_label

dataset = dataset.map(template_dataset)
print("Performing model inference...")
dataset = dataset.map(lambda x: {"predicted_emotion": generate_and_decode(x['text']),"true_emotion":x['emotion']})
print("Model inference done.")
train_test_split = dataset.train_test_split(test_size=0.05,seed=42)
train_set = train_test_split["train"]
test_set = train_test_split["test"]


client = OpenAI()

def get_embeddings(text):
    response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

emotions = set()
for _, example in enumerate(dataset):
  emotions.add(example['emotion'])

print("Generating embeddings...")
emotion_map = dict()
for emotion in emotions:
    emotion_map[emotion] = get_embeddings(emotion) #caching values
print("Embeddings generated")

def cosine_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

print("Calculating accuracy...")
# Calculate the accuracy
total = 0
for i,row in enumerate(train_set):
    pred_emotion = row['predicted_emotion']
    if pred_emotion in emotion_map:
        embedding2 = emotion_map[pred_emotion]
    else:
        embedding2 = get_embeddings(pred_emotion)
    similarity = cosine_angle(emotion_map[row['emotion']],embedding2)
    total+=similarity

accuracy = total / len(train_set)
print(f"Training accuracy: {accuracy * 100:.2f}%")

total = 0
for i,row in enumerate(test_set):
    pred_emotion = row['predicted_emotion']
    if pred_emotion in emotion_map:
        embedding2 = emotion_map[pred_emotion]
    else:
        embedding2 = get_embeddings(pred_emotion)
    similarity = cosine_angle(emotion_map[row['emotion']],embedding2)
    total+=similarity

accuracy = total / len(test_set)
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("Calculating exact match accuracy...")
total = 0
for row in iter(train_set):
    total+=(row['emotion']==row['predicted_emotion'])

accuracy = total / len(train_set)
print(f"Training accuracy (exact match): {accuracy * 100:.2f}%")

total = 0
for row in iter(test_set):
    total+=(row['emotion']==row['predicted_emotion'])

accuracy = total / len(test_set)
print(f"Test accuracy (exact match): {accuracy * 100:.2f}%")



