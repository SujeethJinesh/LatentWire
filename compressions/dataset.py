"""Dataset for compression training."""

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm


class SQuADCompressionDataset(Dataset):
    """SQuAD dataset for compression training."""

    def __init__(self, split='train', num_samples=10000, tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load SQuAD
        dataset = load_dataset('squad', split=split)

        # Sample if needed
        if num_samples and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = dataset.select(indices)

        self.data = []
        for item in tqdm(dataset, desc=f"Processing {split} data"):
            # Format prompt
            prompt = f"Question: {item['question']}\nContext: {item['context']}\nAnswer:"
            answer = item['answers']['text'][0] if item['answers']['text'] else ""

            self.data.append({
                'prompt': prompt,
                'answer': answer,
                'question': item['question'],
                'context': item['context']
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize prompt (NO padding - done in collate_fn)
        prompt_tokens = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Dynamic padding in batch
            return_tensors=None
        )

        # Tokenize answer
        answer_tokens = self.tokenizer(
            item['answer'],
            max_length=32,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        return {
            'input_ids': prompt_tokens['input_ids'],
            'attention_mask': prompt_tokens['attention_mask'],
            'answer_ids': answer_tokens['input_ids'],
            'answer_mask': answer_tokens['attention_mask'],
            'answer_text': item['answer']
        }

    @staticmethod
    def collate_fn(tokenizer):
        """Create collate function with dynamic padding."""
        def collate(batch):
            # Extract fields
            input_ids = [torch.tensor(item['input_ids']) for item in batch]
            attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
            answer_ids = [torch.tensor(item['answer_ids']) for item in batch]
            answer_masks = [torch.tensor(item['answer_mask']) for item in batch]
            answer_texts = [item['answer_text'] for item in batch]

            # Pad to max length in batch (not global max)
            from torch.nn.utils.rnn import pad_sequence
            input_ids_padded = pad_sequence(input_ids, batch_first=True,
                                           padding_value=tokenizer.pad_token_id)
            attention_masks_padded = pad_sequence(attention_masks, batch_first=True,
                                                 padding_value=0)
            answer_ids_padded = pad_sequence(answer_ids, batch_first=True,
                                            padding_value=tokenizer.pad_token_id)
            answer_masks_padded = pad_sequence(answer_masks, batch_first=True,
                                              padding_value=0)

            return {
                'input_ids': input_ids_padded,
                'attention_mask': attention_masks_padded,
                'answer_ids': answer_ids_padded,
                'answer_mask': answer_masks_padded,
                'answer_text': answer_texts
            }
        return collate
