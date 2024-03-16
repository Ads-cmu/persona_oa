# Model Evaluation

To evaluate the model, follow these steps

1. Clone the repo: `git clone https://github.com/Ads-cmu/persona_oa.git`
2. Change directory to the repo `cd persona_oa`
3. Install required packages `pip install -r requirements.txt`
4. Log in to Huggingface `huggingface-cli login`. Enter your API key when prompted. Run `git config --global credential.helper store` to save login credentials, if they are not saved.
5. Save your OPENAI API key: `export OPENAI_API_KEY="<your-key-here>"`
6. Change the dataset to be loaded (line 9 of `evaluate.py`) if needed
7. Run `python evaluate.py`



