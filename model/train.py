import torch
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.model_dpo import AutoDPOModelForCausalLM
import torch.nn as nn
from datasets import Dataset

# Download the pre-trained model and tokenizer from the Hub
model_name = "Qwen/Qwen1.5-0.5B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize your model class and import the pre-trained model into your class
# Note that if you have a custom module in your class
# You should initialize the weights of this module in the `__init__` function
model_wrapper = AutoDPOModelForCausalLM(pretrained_model=model)

# Define the directory where you want to save the model and tokenizer
save_directory = "/checkpoint/"

# Save the model and tokenizer
# model_wrapper.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)

# We used several datasets for training, this one is an example of train dataset for the format
# You need to load here your own dataset
dpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ],
}

# Convert the dictionary to a datasets.Dataset
train_dataset = Dataset.from_dict(dpo_dataset_dict)

# Configure the training arguments for DPO
training_args = DPOConfig(
    beta=0.1,
)

# Configure the DPO trainer
dpo_trainer = DPOTrainer(
    model_wrapper,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_length=1024,
)

# Train the model with DPO
dpo_trainer.train()

# Save the trained model
dpo_trainer.model.save_pretrained(save_directory)
