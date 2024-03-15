# from openai import OpenAI
import datasets
from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig
import torch
import numpy as np 
import transformers

dataset = datasets.load_dataset("scott-persona/emotion_test_set",split="train")

# Load the fine-tuned model and tokenizer
model_path = "../finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",torch_dtype=torch.float16, load_in_4bit=True)
model.resize_token_embeddings(len(tokenizer))
# print(model)
# model.load_adapter(model_path)


config = LoraConfig.from_pretrained(model_path)

# Load the LoRA model
inference_model = PeftModel.from_pretrained(model, model_path) 

# model = AutoModelForCausalLM.from_config(config,torch_dtype=torch.float16, load_in_4bit=True)

def generate_and_decode(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    # Get the last generated token ID
    predicted_token_id = logits[:, 0, :].argmax(dim=-1)
    # Decode the token ID to text
    predicted_label = tokenizer.decode(predicted_token_id).strip()
    print(tokenizer.decode(logits.argmax(dim=-1)[0]))
    return predicted_label


total = 0
for i,row in enumerate(dataset):
    pred_emotion = generate_and_decode(row['emotion'])
    print(row['emotion'],pred_emotion)
    total+=(row['emotion']==pred_emotion)

accuracy = total / len(dataset)
print(f"Training accuracy: {accuracy * 100:.2f}%")




# client = OpenAI()

# def get_embeddings(text):
#     # Generate embeddings using the "ada-2" model
#     response = client.embeddings.create(
#     input="Your text string goes here",
#     model="text-embedding-ada-002"
#     )
#     # Extract the embedding vector from the response
#     embedding = response.data[0].embedding
#     return embedding

# emotions = set()
# for _, example in enumerate(dataset):
#   emotions.add(example['emotion'])

# emotion_map = dict()
# for emotion in emotions:
#     emotion_map[emotion] = get_embeddings(emotion) #caching values
    
# Apply the prediction function to the training set
# dataset = dataset.map(lambda x: {"predicted_emotion": generate_and_decode(x['text']),"true_emotion":x['emotion']})

# # Calculate the accuracy
# total = 0
# for i,row in enumerate(dataset):
#     pred_emotion = row['predicted_emotion']
#     if pred_emotion in emotion_map:
#         embedding2 = emotion_map[pred_emotion]
#     else:
#         embedding2 = get_embeddings(pred_emotion)
#     similarity = np.dot(emotion_map[row['emotion']],embedding2)
#     total+=similarity

# accuracy = total / len(dataset)
# print(f"Training accuracy: {accuracy * 100:.2f}%")


