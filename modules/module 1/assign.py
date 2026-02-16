import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix

# Load data (assuming movie_data.csv is in the current directory)
df = pd.read_csv('datasets/movie_data.csv')


# def clean_html(text):
#     # Specifically removes <br />, <i>, </i>, <em>, </em> and their variants
#     clean = re.sub(r'<(br|i|em)\s*/?>|</(i|em)>', ' ', str(text).lower())
#     return clean

HTML_TAGS = [
    r'<\s*br\s*/?\s*>',
    r'<\s*/?\s*i\s*>',
    r'<\s*/?\s*em\s*>',
]

def strip_known_html(text):
    for pat in HTML_TAGS:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    return text

token_pattern= r'(?u)\b[a-zA-Z]\w+\b'

df['clean_review'] = df['review'].apply(strip_known_html)
df['tokens_task1'] = df['clean_review'].apply(lambda x: regexp_tokenize(x.lower(), token_pattern))

# Task 2: Sentence counts (No stop-word removal)
df['sentences'] = df['clean_review'].apply(sent_tokenize)
sent_counts = df.groupby('sentiment')['sentences'].apply(lambda sents: sum(len(s) for s in sents))
print(f"Task 2 - Total Sentences:\n{sent_counts}\n")

# Task 3: Nested Tokenization (Review -> Sentence -> Words)
df['nested_tokens'] = df['sentences'].apply(lambda sents: [regexp_tokenize(s.lower(), token_pattern) for s in sents])

for s in [0, 1]:
    group = df[df['sentiment'] == s]['nested_tokens']
    word_counts = [sum(len(sent) for sent in rev) for rev in group]
    print(f"Sentiment {s}: Total Words={sum(word_counts)}, Min={min(word_counts)}, Max={max(word_counts)}")

class Markov:  # Uses a sparse matrix to handle probabilities
    def __init__(self, order=1):
        self.voc, self.vocindex, self.vocindexrev = None, None, None
        self.conds, self.condsindex, self.condsindexrev = None, None, None
        self.Pjoint = None  # Joint probabilities
        self.Pvoc = None  # marginal P for vocabulary
        self.Pconds = None  # marginal P for conditionings
        self.Pconditional = None  # Conditional probabilities, Markov model
        self.order = order

    def get_vocabulary(self, _toks):  # input as list of tokens, 1-gram
        voc = set()
        for _ in _toks:
            voc.update(_)
        self.voc = sorted(voc)
        self.vocindex = {v:i for i, v in enumerate(self.voc)}
        self.vocindexrev = {i:v for i, v in enumerate(self.voc)}
    
    def get_conds(self, _toks):  # conditioning events, n-1-gram
        conds = set()
        for tok in _toks:
            for i in range(len(tok)-self.order):  # boundary condition
                cond = " ".join(tok[i:i+self.order])
                conds.update([cond])
        self.conds = sorted(conds)
        self.condsindex = {cond:i for i, cond in enumerate(self.conds)}
        self.condsindexrev = {i:cond for i, cond in enumerate(self.conds)}
    
    def get_counts(self, _toks):  # build the P
        M, N = len(self.vocindex), len(self.condsindex)
        pc = lil_matrix((N,M), dtype=np.float32)
        # pc = np.zeros((N,M), dtype=np.float32)
        for tok in _toks:
            for i in range(len(tok)-self.order-1):  # boundary condition
                cond = " ".join(tok[i:i+self.order])
                voc = tok[i+self.order]
                i, j = self.condsindex[cond], self.vocindex[voc]
                pc[i,j] += 1
        return pc

    def fit(self, _toks):
        self.get_vocabulary(_toks)
        self.get_conds(_toks)
        pc = self.get_counts(_toks)
        # joint P, make it into probabilities
        self.Pjoint = pc / np.sum(pc)
        # marginal P for vocabulary and conds
        self.Pvoc = np.array(np.sum(self.Pjoint,axis=0)).squeeze()
        self.Pconds = np.array(np.sum(self.Pjoint,axis=1)).squeeze()
        # conditional P
        sm = np.array(pc.sum(axis=1)).squeeze()
        sm[sm==0] = 1  # handle divide by zero
        self.Pconditional = lil_matrix(pc / sm[:,None])
        # sanity
        print(f'Size of vocabulary={len(self.voc)}, N={len(self.conds)}, Pjoint.shape={self.Pjoint.shape}')
        return self

markov = Markov(order=1).fit(df['tokens_task1'])

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
stop_words = set(stopwords.words('english'))

def clean_for_markov(text):
    text = strip_known_html(text)
    # Remove symbols and numbers (keep only a-z)    
    tokens = regexp_tokenize(text.lower(), token_pattern)
    return [w for w in tokens if w not in stop_words]

# Prepare cleaned data
df['tokens_markov'] = df['review'].apply(clean_for_markov)
tokens_neg = df[df['sentiment'] == 0]['tokens_markov'].tolist()
tokens_pos = df[df['sentiment'] == 1]['tokens_markov'].tolist()

# Task 4: Order 1 Models
m0_ord1 = Markov(order=1).fit(tokens_neg)
m1_ord1 = Markov(order=1).fit(tokens_pos)

# Task 6 & 8: Order 3 Models (required for 3-word sequences)
m0_ord3 = Markov(order=3).fit(tokens_neg)
m1_ord3 = Markov(order=3).fit(tokens_pos)


# Task 5
print(f"P('good' | Neg): {m0_ord1.Pvoc[m0_ord1.vocindex['good']]:.6f}")
print(f"P('good' | Pos): {m1_ord1.Pvoc[m1_ord1.vocindex['good']]:.6f}")


def get_top_next_words(model, context, n=3):
    """Returns the top n words and their probabilities for a given context."""
    # Ensure context is cleaned and tokenized the same way as training data
    # Note: For Task 6/8, the context must match the model's order.
    if context not in model.condsindex:
        return None
    
    row_idx = model.condsindex[context]
    # Convert sparse row to dense to find the top probabilities
    row_data = model.Pconditional[row_idx, :].toarray().flatten()
    
    # Get indices of the highest probabilities
    top_indices = np.argsort(row_data)[-n:][::-1]
    
    results = []
    for idx in top_indices:
        if row_data[idx] > 0:
            results.append((model.vocindexrev[idx], row_data[idx]))
    return results

def get_task_results(model, context, target=None, top_n=3):
    """
    Helper to handle both probability lookup (Task 6) 
    and word generation (Tasks 7 & 8).
    """
    # 1. Prepare the context: Lowercase and filter out stop words to match training data
    # Note: 'one', 'best', 'movies', 'worst', 'ever' are generally NOT stop words 
    # in NLTK, but we run the filter to be safe.
    tokens = regexp_tokenize(context.lower(), token_pattern)
    clean_context = " ".join([w for w in tokens if w not in stop_words])
    
    # Check if context exists in this specific model
    if clean_context not in model.condsindex:
        return f"Context '{clean_context}' not found in this model."

    row_idx = model.condsindex[clean_context]
    
    # Task 6 Logic: Specific target probability
    if target:
        if target not in model.vocindex:
            return f"Target word '{target}' not in vocabulary."
        col_idx = model.vocindex[target]
        prob = model.Pconditional[row_idx, col_idx]
        return prob

    # Task 7/8 Logic: Generate top N words
    # Convert sparse row to dense array
    probs = model.Pconditional[row_idx, :].toarray().flatten()
    top_indices = np.argsort(probs)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if probs[idx] > 0:
            results.append(f"{model.vocindexrev[idx]} ({probs[idx]:.6f})")
    return results

# --- Task 6: P("ever" | "one best movies") ---
print("--- Task 6 ---")
target = "ever"
ctx6 = "one best movies"
print(f"Sentiment 0: {get_task_results(m0_ord3, ctx6, target)}")
print(f"Sentiment 1: {get_task_results(m1_ord3, ctx6, target)}")

# --- Task 7: Generate 3 words after "worst" ---
print("\n--- Task 7 (Order 1) ---")
ctx7 = "worst"
print(f"Sentiment 0 (Top 3 after 'worst'): {get_task_results(m0_ord1, ctx7)}")
print(f"Sentiment 1 (Top 3 after 'worst'): {get_task_results(m1_ord1, ctx7)}")

# --- Task 8: Generate 3 words after "worst movie ever" ---
print("\n--- Task 8 (Order 3) ---")
ctx8 = "worst movie ever"
print(f"Sentiment 0 (Top 3 after 'worst movie ever'): {get_task_results(m0_ord3, ctx8)}")
print(f"Sentiment 1 (Top 3 after 'worst movie ever'): {get_task_results(m1_ord3, ctx8)}")