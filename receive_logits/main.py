import gc
import os
import math
import torch
import huggingface_hub
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

huggingface_hub.login('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')
load_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=load_config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

os.environ["DATA_DIR"] = "data/dolma"
dataset = load_dataset("allenai/dolma", split="train", name="v1_6-sample").shuffle(seed=42)


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, ids, masks):
        super(LMDataset).__init__()
        self.ids = ids
        self.masks = masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {'input_ids': self.ids[idx], 'attention_mask': self.masks[idx]}

def make_dataset(dataset, chunk_len=128):
    chunk_len += 1
    real_tokens, pad_tokens = 0, 0
    ids = []
    masks = []
    for text in tqdm(dataset):
        tokens = tokenizer(text + tokenizer.eos_token, return_tensors='pt')['input_ids'][0]
        token_len = len(tokens)
        if token_len <= 2:
            continue
        pad_len = math.ceil(token_len / chunk_len) * chunk_len - token_len
        real_tokens += token_len
        pad_tokens += pad_len
        pads = torch.zeros(pad_len, dtype=int)
        masks.append(torch.cat([torch.ones(token_len, dtype=int), pads]).view(-1, chunk_len))
        ids.append(torch.cat([tokens, pads]).view(-1, chunk_len))
    ids = torch.cat(ids)
    masks = torch.cat(masks)
    print(f"Real tokens: {real_tokens}\tPad tokens: {pad_tokens}")
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
            input_ids=batch['input_ids'][:, :-1],
            attention_mask=batch['attention_mask'][:, :-1],
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


SIZE = 200000
N = 100
K = 100
L = 5000
ids, masks = make_dataset(dataset[:SIZE]['text'])
rand_idx = torch.randperm(ids.shape[0])
ids = ids.index_select(0, rand_idx)
masks = masks.index_select(0, rand_idx)
N = min(N, ids.shape[0] // L)
for i in range(N):
    print(f"Processing chunk: {i}/{N}")
    train_dataset = LMDataset(ids[i*L:(i+1)*L].clone(), masks[i*L:(i+1)*L].clone())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
    values, indices = eval_model(model, train_dataloader, k=K)
    os.makedirs(f"/root/logits{K}", exist_ok=True)
    save_results(values, indices, train_dataset, name=f'logits{K}/chunk{i}')
print(f"Processing chunk: {N}/{N}")
