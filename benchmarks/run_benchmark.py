import gc
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
import tensorflow_datasets as tfds
import datasets

DEVICE = 'cuda'
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id
load_config = BitsAndBytesConfig(load_in_8bit=True)

from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

def batch_to_device(batch, device='cuda'):
    return {k: v.to(device) for k, v in batch.items()}

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

class BenchDataset(Dataset):
    def __init__(self, dataset: Dataset, prompt, examples_cnt, options, metric=None):
        super(BenchDataset).__init__()
        examples = dataset.take(examples_cnt)
        p_ex = prompt + "{0[label]}\n"
        if len(examples) > 0:
            prompt = "Examples:\n" + ''.join(list(map(lambda x: p_ex.format(x|{'label':options[x['label']]}), examples))) + "\nTask:\n" + prompt
        else:
            prompt = "Task:\n" + prompt
        print(prompt)
        dataset = dataset.skip(examples_cnt).map(lambda row: {"prompt": prompt.format(row)})
        tokens = tokenizer(dataset['prompt'], return_tensors='pt', padding=True)
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = dataset['label']
        self.options = tokenizer(options, return_tensors='pt')['input_ids'][:, 1]
        self.metric = metric

    def __getitem__(self, index):
        return {'input_ids': self.input_ids[index], 'attention_mask': self.attention_mask[index], 'label': self.labels[index]}

    def __len__(self):
        return self.input_ids.size(0)
    
    def fine_tune(self, model, epochs=10, batch_size=32, lr=3e-4, shedule=False):
        model = lora_model(model)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, lr/5)
        dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size)
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                batch = batch_to_device(batch, DEVICE)
                logits = model(
                    input_ids = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                ).logits
                ids = (batch['attention_mask'].sum(dim=1) - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, logits.size(2))
                logits = logits.gather(1, ids)[:, 0, :]
                loss = criterion(logits[:, self.options], batch['label'])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                gc.collect()
                torch.cuda.empty_cache()
            if shedule:
                scheduler.step()
            print(f"Epoch {epoch}. Loss {total_loss / len(dataloader)}")
        self.evaluate(model, batch_size)
        return model
        
    
    def evaluate(self, model, batch_size=16):
        model.eval()
        preds = []
        dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size)
        for batch in tqdm(dataloader):
            batch = batch_to_device(batch, DEVICE)
            logits = model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
            ).logits
            ids = (batch['attention_mask'].sum(dim=1) - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, logits.size(2))
            logits = logits.gather(1, ids)[:, 0, :].detach().cpu()
            preds.extend(logits[:, self.options].argmax(dim=1))
            batch = None
            logits = None
            gc.collect()
            torch.cuda.empty_cache()
        print(torch.sum(torch.tensor(self.labels) == torch.tensor(preds))/len(self))
        return self.labels, preds
    
import random


def axb_dataset(examples_cnt=0):
    dataset = load_dataset("super_glue", "axb", split='test').shuffle(seed=0)
    prompt = """Question: Do sentences '{0[sentence1]}' and '{0[sentence2]}' mean the same thing? Answer:"""
    return BenchDataset(
        dataset,
        prompt,
        examples_cnt,
        ['Yes', 'No']
    )

def sst2_dataset(examples_cnt=0, split='validation'):
    dataset = load_dataset("stanfordnlp/sst2", split=split).shuffle(seed=0)
    prompt = """{0[sentence]} Is the review positive? Answer:"""
    return BenchDataset(
        dataset,
        prompt,
        examples_cnt,
        ['No', 'Yes']
    )

def boolq_dataset(examples_cnt=0, split='validation'):
    dataset = load_dataset("super_glue", "boolq", split=split).shuffle(seed=0).map(lambda x: {'passage':x['passage'][:500]})
    if split == 'validation':
        dataset = dataset.take(1000)
    prompt = """{0[passage]}\nQuestion: {0[question]}?\nAnswer:"""
    return BenchDataset(
        dataset,
        prompt,
        examples_cnt,
        ['No', 'Yes']
    )

def qnli_dataset(examples_cnt=0, split='validation'):
    dataset = tfds.load('glue/qnli', split=split)
    labels = [data['label'] for data in dataset]
    questions = [data['question'].numpy().decode('utf-8') for data in dataset]
    sentences = [data['sentence'].numpy().decode('utf-8') for data in dataset]
    dataset = datasets.Dataset.from_dict({
        'label': labels,
        'question': questions,
        'sentence': sentences,
    }).shuffle(seed=0).map(lambda x: {'sentence':x['sentence'][:200]})
    if split == 'validation':
        dataset = dataset.take(1000)
    prompt = """Request: {0[question]}\nResponse: {0[sentence]}\nQuestion: Is it suitable Response for the Request? Answer:"""
    return BenchDataset(
        dataset,
        prompt,
        examples_cnt,
        ['Yes', 'No']
    )

def irregular_forms_dataset(examples_cnt=0):
    dataset = concatenate_datasets([
        load_dataset("nyu-mll/blimp", name='irregular_past_participle_adjectives', split='train'),
        load_dataset("nyu-mll/blimp", name='irregular_past_participle_verbs', split='train'),
        load_dataset("nyu-mll/blimp", name='irregular_plural_subject_verb_agreement_1', split='train'),
        load_dataset("nyu-mll/blimp", name='anaphor_gender_agreement', split='train'),
        load_dataset("nyu-mll/blimp", name='anaphor_number_agreement', split='train'),
        load_dataset("nyu-mll/blimp", name='determiner_noun_agreement_1', split='train'),
        load_dataset("nyu-mll/blimp", name='irregular_plural_subject_verb_agreement_2', split='train'),
        load_dataset("nyu-mll/blimp", name='irregular_plural_subject_verb_agreement_2', split='train'),
        load_dataset("nyu-mll/blimp", name='irregular_plural_subject_verb_agreement_2', split='train'),
    ])
    dataset = dataset.shuffle(seed=10).take(1000)
    
    def mapper(x):
        good_split = x['sentence_good'].split(' ')
        bad_split = x['sentence_bad'].split(' ')
        vals = []
        sent = []
        for i in range(len(good_split)):
            if good_split[i] == bad_split[i]:
                sent.append(good_split[i])
            else:
                vals.append(good_split[i])
                vals.append(bad_split[i])
                sent.append('[PAD]')
        ref = vals[0]
        label = random.randint(0, 1)
        if label == 1:
            vals[0], vals[1] = vals[1], vals[0]
        return {
            'sent': ' '.join(sent),
            'good': vals[0],
            'bad': vals[1],
            'label': label,
            'ref': ref,
        }

    dataset = dataset.map(mapper)
    prompt = """{0[sent]}
Select the correct replacement for [PAD]: A) {0[good]} B) {0[bad]} Answer:"""

    return BenchDataset(
        dataset,
        prompt,
        examples_cnt,
        ['A', 'B']
    )
    
for func in [sst2_dataset, boolq_dataset, qnli_dataset]:
    random.seed(0)
    dataset = func()
    train_dataset = func(0, 'train')
    model = LlamaForCausalLM.from_pretrained('/root/llama_distillation/experiments/models/baseline__distill_only').to(DEVICE)
    model = train_dataset.fine_tune(model)
    dataset.evaluate(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()


