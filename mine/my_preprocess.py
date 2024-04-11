import numpy as np
import re

data_path = "../spa-eng/spa.txt"

with open(data_path, 'r', encoding='utf-8') as f:
     lines = f.read().split('\n')
     
input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()
     
