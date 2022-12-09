import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import re
import pandas as pd

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


##GPU 사용 시
device = torch.device("cuda:0")
print(f'device : {device}')
## Setting parameters
max_len = 64
batch_size = 128
warmup_ratio = 0.1
max_grad_norm = 1
log_interval = 200

bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
class BERTDataset_test(Dataset):
    def __init__(self, dataset,bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([" ".join(str(dataset.iloc[i]['sentence']))]) for i in range(len(dataset))]
        self.labels = [np.int32(dataset.iloc[i]['label']) for i in range(len(dataset))]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    


pred_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
pred_model.load_state_dict(torch.load('/home/mglee/VSCODE/git_folder/BERT/Sentiment/Kobert_Sentiment_e5_state_dict_221018.pt'))  # state_dict를 불러 온 후, 모델에 저장

def main(body):
    dataset = pd.DataFrame(body)
    dataset['label'] = 0
    dataset.columns = ['sentence', 'label']
    print("BERT dataset creating......")
    comment_bert = BERTDataset_test(dataset, tok, max_len, True, False)
    pred_dataloader = torch.utils.data.DataLoader(comment_bert, batch_size = batch_size, num_workers = 8)
    out_lst = []
    
    print(("BERT Classifier Start......"))
    for (token_ids, valid_length, segment_ids, _) in tqdm(pred_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = pred_model(token_ids, valid_length, segment_ids)
        out_lst.append(out.data.cpu())
        max_vals, max_indices = torch.max(out, 1)
    pred = []
    print("Predicting......")
    for batch in out_lst:
        for item in batch:
            pred.append(np.argmax(item.numpy()))
        
    return pred
