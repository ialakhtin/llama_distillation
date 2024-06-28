import gc
import os
import math
import torch
import huggingface_hub
from tqdm import tqdm
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

huggingface_hub.login('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')
load_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=load_config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
dataset = load_from_disk('/root/dolma_dataset_500k')


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, ids, masks):
        super(LMDataset).__init__()
        self.ids = ids
        self.masks = masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {'input_ids': self.ids[idx], 'attention_mask': self.masks[idx]}

def make_dataset(dataset, chunk_len=256):
    real_tokens, pad_tokens, skipped = 0, 0, 0
    ids = []
    masks = []
    eol_token = 13
    bos_token = 1
    eos_token = 2
    is_same_line = False
    for row in tqdm(dataset):
        if row['source'] == 'stack-dedup':
            continue
        text = row['text']
        lines = text.split('\n')
        chunk = [bos_token]
        for line in lines:
            sents = sent_tokenize(line)
            for sent in sents:
                tokens = tokenizer.encode(sent, add_special_tokens=False)
                if len(chunk) > 1 and not is_same_line and len(chunk) + len(tokens) < chunk_len - 1:
                    chunk.append(eol_token)
                is_same_line = True
                if len(chunk) + len(tokens) < chunk_len:
                    chunk.extend(tokens)
                    continue
                if len(chunk) > 1:
                    chunk.append(eos_token)
                    pad_len = chunk_len - len(chunk)
                    pad_tokens += pad_len
                    real_tokens += len(chunk) - 2
                    pads = [0]*pad_len
                    ids.append(chunk + pads)
                    masks.append([1] * len(chunk) + pads)
                chunk = [bos_token]
                if len(tokens) >= chunk_len - 1 or len(tokens) == 0:
                    skipped += len(tokens)
                    continue
                chunk.extend(tokens)
            is_same_line = False
    ids = torch.tensor(ids)
    masks = torch.tensor(masks)
    print(f"Real tokens: {real_tokens}\tPad tokens: {pad_tokens}\tSkipped: {skipped}")
    return ids, masks


def batch_to_device(batch, device='cuda'):
    return {k: v.to(device) for k, v in batch.items()}

def eval_model(model, dataloader, k=10):
    model.eval()
    values = []
    indices = []
    for i, batch in enumerate(tqdm(dataloader)):
        batch = batch_to_device(batch)
        out = model.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        ).loss['logits'].detach().cpu().clone()
        topk = torch.topk(out, k, dim=2)
        values.append(topk.values)
        indices.append(topk.indices)
        batch = batch_to_device(batch, device='cpu')
        gc.collect()
        torch.cuda.empty_cache()
    values = torch.cat(values)
    indices = torch.cat(indices)
    return values, indices 


def save_results(values, indices, dataset, name='result'):
    os.makedirs(f"/root/{name}", exist_ok=True)
    torch.save(values, f"/root/{name}/values.pt")
    torch.save(indices, f"/root/{name}/indices.pt")
    torch.save(dataset.ids, f"/root/{name}/input_ids.pt")
    torch.save(dataset.masks, f"/root/{name}/attention_masks.pt")


SIZE = 250000
N = 80
K = 100
L = 5000
ids, masks = make_dataset(dataset.take(SIZE))
rand_idx = torch.randperm(ids.shape[0])
ids = ids.index_select(0, rand_idx)
masks = masks.index_select(0, rand_idx)
N = min(N, ids.shape[0] // L)
for i in range(N):
    print(f"Processing chunk: {i}/{N}")
    train_dataset = LMDataset(ids[i*L:(i+1)*L].clone(), masks[i*L:(i+1)*L].clone())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    values, indices = eval_model(model, train_dataloader, k=K)
    os.makedirs(f"/root/logits{K}", exist_ok=True)
    save_results(values, indices, train_dataset, name=f'logits{K}/chunk{i}')
print(f"Processing chunk: {N}/{N}")
