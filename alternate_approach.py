import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def fetch_embedding(text):
  encoded_input = tokenizer(text, return_tensors='pt')
  # Generate embeddings
  with torch.no_grad():
      outputs = model(**encoded_input)
  embeddings = outputs.last_hidden_state
  mean_embedding = embeddings.mean(dim=1)
  return mean_embedding


# Load the dataset
dataset = datasets.load_dataset("scott-persona/emotion_test_set", split="train")
# Encode labels
text_column = "situation"
label_column = "emotion"
emotions = set()
for _, example in enumerate(dataset):
  emotions.add(example[label_column])

label_encoder = LabelEncoder(len(emotions))
dataset = dataset.map(lambda examples: {"labels": label_encoder.fit_transform(examples[label_column])}, batched=True) 
  
train_test_split = dataset.train_test_split(test_size=0.05, seed=42)
train_set = train_test_split["train"]
test_set = train_test_split["test"]

# Initialize BERT tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = RobertaModel.from_pretrained('roberta-base')

# Tokenize the text
def tokenize(batch):
    return tokenizer(batch[text_column], padding=True, truncation=True, max_length=512, return_tensors="pt")

train_set = train_set.map(tokenize, batched=True, batch_size=len(train_set))
test_set = test_set.map(tokenize, batched=True, batch_size=len(test_set))

# Convert to PyTorch tensors and create dataloaders
train_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16)

# Define the classifier model
class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Instantiate the model
num_classes = 27
model = EmotionClassifier(bert_model, num_classes)

# Training settings
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):  # Loop over the dataset multiple times
    total_loss = 0.0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Simple accuracy check (for demonstration purposes, consider using a more detailed evaluation)
model.eval()
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy}")

