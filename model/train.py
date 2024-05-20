import torch
import json
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.utils.data import Dataset


class DPO_Dataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(model, dataloader, optimizer, reference_model, tokenizer, num_epochs=3):
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_loss = 0
        start_time = time.time()
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            optimizer.zero_grad()

            chosen_logps, rejected_logps = model.get_logprobs(batch, tokenizer)
            reference_chosen_logps, reference_rejected_logps = reference_model.get_logprobs(batch, tokenizer)

            rewards = model.prediction_step_reward(chosen_logps,
                                                   rejected_logps,
                                                   reference_chosen_logps,
                                                   reference_rejected_logps)

            chosen_rewards = rewards["chosen_rewards"]
            rejected_rewards = rewards["rejected_rewards"]

            dpo_loss = - F.logsigmoid(chosen_rewards - rejected_rewards)
            dpo_loss.backward()
            optimizer.step()

            total_loss += dpo_loss.item()

        # Calculate time taken for epoch
        epoch_time = time.time() - start_time
        tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {total_loss / len(dataloader):.4f} - Time: {epoch_time:.2f} s")
