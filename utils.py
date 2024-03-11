from nltk.corpus import stopwords
import re
from transformers import BertTokenizer
import torch
import random
import numpy as np
from tabulate import tabulate

sw = stopwords.words('english')

def clean_text(text):
        
        text = text.lower()
        text = re.sub(r"[^a-zA-Z?.!¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!")
        text = re.sub(r"http\S+", "",text) #Removing URLs 
        html=re.compile(r'<.*?>') 
        text = html.sub(r'',text) #Removing html tags
        punctuations = ',@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'

        for p in punctuations:
            text = text.replace(p,'') #Removing punctuations


        text = [word.lower() for word in text.split() if word.lower() not in sw]        
        text = " ".join(text) #removing stopwords
        


        
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text) #Removing emojis
        
        return text

def tokenize(sentences, max_len):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            truncation = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
        )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

def print_rand_sentence_encoding(sentences, input_ids, attention_masks):
    '''Displays tokens, token IDs and attention mask of a random text sample'''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    '''get a random row from dataset that is a pandas dataframe'''
    index = random.randint(0, len(sentences) - 1)
    # select row index from dataset
    sentence = list(sentences)[index]
    tokens = tokenizer.tokenize(tokenizer.decode(input_ids[index]))
    token_ids = [i.numpy() for i in input_ids[index]]
    attention = [i.numpy() for i in attention_masks[index]]

    table = np.array([tokens, token_ids, attention]).T
    print(sentence)
    print(tabulate(table, 
                    headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                    tablefmt = 'fancy_grid'))

def print_report(training_stats):
    '''Prints the training report'''
    headers = ['Epoch', 'Valid. Accur.', 'Training Time', 'Valid. Time']
    table = [[i['epoch'], i['Valid. Accur.'], i['Training Time'], i['Valid. Time']] for i in training_stats]
    print(tabulate(table, headers, tablefmt='fancy_grid'))