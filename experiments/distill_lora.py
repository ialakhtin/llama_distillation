import gc
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model


DEVICE = 'cuda'
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def lora_model(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.linear.Linear):
            lora_module_names.add(name.split('.')[-1])
    config = LoraConfig(
        r=8,  #attention heads
        lora_alpha=8,  #alpha scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"],  #gonna train all
        lora_dropout=0.5,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM", #for Decoder models like GPT Seq2Seq for Encoder-Decoder models like T5
    )
    model = get_peft_model(model, config)
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return model

class LogitsDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        super(LogitsDataset).__init__()
        self.ids = torch.load(f"{dir_path}/input_ids.pt")
        self.masks = torch.load(f"{dir_path}/attention_masks.pt")
        self.indices = torch.load(f"{dir_path}/indices.pt")
        self.values = torch.load(f"{dir_path}/values.pt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.ids[idx],
            'attention_mask': self.masks[idx],
            'indices': self.indices[idx],
            'values': self.values[idx],
        }


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


class Trainer:
    def __init__(self, model, config, criterion):
        self.model = model
        self.T = config['T']
        self.n_epoch = config['n_epoch']
        self.batch_size = config['batch_size']
        self.optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 200, config['optimizer']['lr']/10)
        self.criterion = criterion

    def get_loss(self, batch):
        mask = batch['attention_mask']
        ids = batch['input_ids']
        labels = ids - 100 * (1 - mask)
        out = self.model.forward(
            input_ids=ids,
            attention_mask=mask,
            labels=labels
        )
        targets = torch.softmax(batch['values']/self.T, dim=2)
        sm = torch.log_softmax(out.logits/self.T, dim=2)
        preds = sm.gather(2, batch['indices'])
        return self.criterion(preds, targets * mask.reshape(*mask.shape, 1)) / mask.sum()
        # return (out.loss + self.criterion(preds, targets * mask.reshape(*mask.shape, 1)) / mask.sum()) / 2

    def process_dataloader(self, dataloader, chunk_id):
        total_loss = 0
        for batch in tqdm(dataloader, f"chunk {chunk_id}"):
            batch = batch_to_device(batch, DEVICE)
            self.optimizer.zero_grad()
            loss = self.get_loss(batch)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
        total_loss /= len(dataloader)
        print(f"Chunk {chunk_id}.\tLoss {total_loss}")
        
        outputs = self.model.generate(
            **batch_to_device(tokenizer("What", return_tensors="pt"), DEVICE),
            max_length=100,
        )
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        return total_loss

    def train_model(self, chunk_dir, n_chunks):
        model.train()
        for epoch in range(self.n_epoch):
            print(f"Start epoch {epoch}")
            total_loss = 0
            for i in range(n_chunks):
                dataset = LogitsDataset(f'{chunk_dir}/chunk{i}')
                dataloader = DataLoader(dataset, batch_size=self.batch_size)
                total_loss += self.process_dataloader(dataloader, i)
            print(f"Epoch {epoch}.\tLoss: {(total_loss / n_chunks):0,.2f}")
            self.model.save_pretrained(f'/root/llama_distillation/experiments/models/{exp_name}_{suffix}')



exp_name = 'tiny_llama'
suffix = ''
with open(f'/root/llama_distillation/experiments/{exp_name}.json') as f:
    config = json.load(f)
model = LlamaForCausalLM.from_pretrained(model_name).to(DEVICE)
model = lora_model(model)

outputs = model.generate(
    **batch_to_device(tokenizer(["Is", "What", "Who", "Hello", "I"], return_tensors="pt"), DEVICE),
    max_length=100,
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# print(f"Number of parameters: {round(sum(p.numel() for p in model.parameters())/1e6)}M\tdevice: {DEVICE}")
trainer = Trainer(model, config, torch.nn.KLDivLoss(reduction="sum"))
trainer.train_model('/root/logits100', 80)
model.save_pretrained(f'/root/llama_distillation/experiments/models/{exp_name}{suffix}')