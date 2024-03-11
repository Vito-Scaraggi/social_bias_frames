import pandas as pd
import os
from utils import clean_text, tokenize
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

class SBICDataset:

    #https://huggingface.co/datasets/social_bias_frames
    def __init__(self):
        self.num_labels = 4
        self.labels = {
            'race' : 0,
            'gender' : 1,
            'culture' : 2,
            'other' : 3
        }

        if not os.path.exists('dataset/train.csv') or not os.path.exists('dataset/val.csv') or not os.path.exists('dataset/test.csv'):
            self.train_data = pd.read_csv('dataset/SBIC.v2.trn.csv')
            self.test_data = pd.read_csv('dataset/SBIC.v2.tst.csv')
            self.val_data = pd.read_csv('dataset/SBIC.v2.dev.csv')

            self.train = self.preprocess_data(self.train_data, 'train')
            self.val = self.preprocess_data(self.val_data, 'val')
            self.test = self.preprocess_data(self.test_data, 'test')
            
        else:
            self.train = pd.read_csv('dataset/train.csv')
            self.val = pd.read_csv('dataset/val.csv')
            self.test = pd.read_csv('dataset/test.csv')
        
        total_set = pd.concat([self.train, self.val, self.test])
        self.max_len, self.avg_len = self.get_len_info(total_set['text'])

    def preprocess_data(self, data, type):
        # select only 'text' and 'label' columns
        data = data[['post', 'offensiveYN', 'targetCategory']]
        # drop rows with NaN values
        data = data.dropna()
        # select rows such that 'offensiveYN' is 1.0
        data = data[data['offensiveYN'] == 1.0]
        #print(data['targetCategory'].value_counts())
        
        # create new column 'label' based on 'targetCategory'
        data['label'] = data.apply(lambda row: self.labels.get(row['targetCategory'], 3), axis=1)
        data = data.rename(columns={'post': 'text'})
        data = data[['text', 'label']]
        # clean text
        data['text'] = data['text'].apply(lambda x: clean_text(x))
        # drop rows with empty text
        data = data[data['text'] != '']
        data = data.dropna()
        # save data to csv
        data.to_csv(f'dataset/{type}.csv', index=False)
        return data

    def get_data(self, type='train'):
        data = self.train if type == 'train' else ( self.test if type == 'test' else self.val )
        return data

    def get_tokenized_data(self, type='train'):
        data = self.train if type == 'train' else ( self.test if type == 'test' else self.val )
        sentences = list(data['text'])
        labels = list(data['label'])
        # convert labels to one-hot encoding
        #labels = [ [1 if i == label else 0 for i in range(self.num_labels)] for label in labels ]
        labels = torch.tensor(labels)
        MAX_LEN = 64
        input_ids, attention_masks = tokenize(sentences, MAX_LEN)
        return TensorDataset(input_ids, attention_masks, labels)

    def get_len_info(self, sentences):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        freqlist = {}
        # For every sentence...
        for sent in list(sentences):

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(sent, add_special_tokens=True)

            if freqlist.get(len(input_ids)) is None:
                freqlist[len(input_ids)] = 0
            
            freqlist[len(input_ids)] += 1

        max_len = max(freqlist.keys())
        avg_len = sum([k*v for k,v in freqlist.items()]) / sum(freqlist.values())
        
        
        for x in [16,32,64,128,256]:
            tot = 0
            for k,v in freqlist.items():
                if k > x:
                    tot += v
            print(f'Number of sentences with more than {x} tokens: {tot}')
         
        return max_len, avg_len

    def __str__(self):
        return f'''\n
        Labels:\n{self.labels}
        Train set: {len(self.train)} samples\n{self.train['label'].value_counts()}
        Validation set: {len(self.val)} samples\n{self.val['label'].value_counts()}
        Test set: {len(self.test)} samples\n{self.test['label'].value_counts()}
        Max token per sentence: {self.max_len}
        Avg token per sentence: {self.avg_len:.2f}'''