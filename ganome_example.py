# extract tokens from sentence using janome

from janome.tokenizer import Tokenizer
import re
import unicodedata
from collections import Counter
import os
import pandas as pd

df = pd.read_csv('data.csv')
raw_text = "\n".join( df['text_col'].dropna().astype(str) )

# Normalize and lightly clean
raw_text = unicodedata.normalize("NFKC", raw)

raw_text = re.sub(r'https?://\S+|www\S+', ' ', raw_text) #remove url
raw_text = re.sub(r'[@#]\S+', ' ', raw_text) #remove email/hashtag
raw_text = re.sub(r'[@【】※★０-９]', raw_text) #remove special characters

#Tokenize
t = Tokenizer()
tokens = []
  for token in tltokenize(raw):
    base = token.base_form
    pos1 = token.part_of_speech.split(',')[0]
    if pos1 in ['名刺', '動詞', '形容詞']:  #Pick the ones you want to keep like 副詞...
      word = base if base != '*' else token.surface
      if len(word)>1:  #Remove 1-char tokens like へ, ん, あ, ... (often noise).
        tokens.append(word)

# Remove ending words.
stopwords = set('''
する ある いる なる こと もの それ これ あれ ため よう さん ところ これら それら そして また ので よう できる ない 的 より など へ に の が を は も と
'''.split() )

# Get tokens.  If you need to get bi-gram, or n-gram, tweak below.
tokens = [w for w in tokens if w not in stopwords]

# Count frequency for each tokens.
freq = Counter(tokens)

# If you need to create wordCrowd, download the font.  https://fonts.google.com/noto/specimen/Noto+Sans+JP
