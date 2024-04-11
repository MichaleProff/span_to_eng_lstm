import numpy as np
import re

data_path = "../spa-eng/spa.txt"

with open(data_path, 'r', encoding='utf-8') as f:
     lines = f.read().split('\n')
     
input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()
     
for line in lines:
     input_doc, target_doc = line.split('\t')[:2]
     input_docs.append(input_doc)

     target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
     target_doc = '<START> ' + target_doc + ' <END>'
     target_docs.append(target_doc)

     for token in re.findall(r"[\w]+|[^\s\w]", input_doc):
          if token not in input_tokens:
               input_tokens.add(token)
     for token in target_doc.split():
          if token not in target_tokens:
               target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

max_encoder_seq_length = max([len(re.findall(r"[\w]+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w]+|[^\s\w]", target_doc)) for target_doc in target_docs])
