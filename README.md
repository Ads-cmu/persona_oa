# Model Evaluation

To evaluate the model, follow these steps

1. Clone the repo: `git clone https://github.com/Ads-cmu/persona_oa.git`
2. Change directory to the repo `cd persona_oa`
3. Install required packages `pip install -r requirements.txt`
4. Log in to Huggingface `huggingface-cli login`. Enter your API key when prompted. Run `git config --global credential.helper store` to save login credentials, if they are not saved.
5. Save your OPENAI API key: `export OPENAI_API_KEY="<your-key-here>"`
6. Change the dataset to be loaded (line 9 of `evaluate.py`) if needed
7. Run `python evaluate.py`

# Model Details
The final model used is the following:
1. Base model: Mistral-7B
2. 27 additional tokens were added to represent the 27 emotions. This had to be done as each emotion was 3-6 tokens in the base tokenizer. 
3. QLoRA finetuning was done to finetune the model. 
4. Some training details: Quantization: 4 bit, r=16, alpha = 64, epochs = 3, optimizer = paged AdamW, LR scheduler = cosine with warmup

# Results
| Metric                | Accuracy     |
|-----------------------|--------------|
| Train Accuracy        | 97.36%       |
| Val Accuracy          | 95.18%       |
| Train Accuracy (Exact Match) | 82.47% |
| Val Accuracy (Exact Match)   | 68%    |


# Alternate Approach
I believe LLMs are overparameterized for this emotion classification task. An alternate approach here could be to finetune a DistillBert model (67M parameters). To try this out, run `python alternate_approach.py`.

| Characteristics    | Mistral    | Bert Finetune | Difference    |
|---------------------|------------|---------------|---------------|
| Parameters          | 7B         | 67M           | 100x smaller  |
| Train Time          | ~2 hours   | ~15 min       | 8x faster     |
| Inference Latency   | 0.54s      | 0.004s        | 135x faster   |
| Accuracy (Val)      | 95.18%     | 93.17%        | 0.8% worse    |



