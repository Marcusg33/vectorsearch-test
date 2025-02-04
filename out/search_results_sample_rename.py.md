## Test 1: Parsing imports with resolvers

**Query**: Which function can find all the import statements in Python code?

**Expected Keywords**: import, resolver, find_imports, ImportResolver

### Rank 1 | Score: 0.7755966

```python
def parse_python_imports(code):
    lines = code.split("\n")
    modules = []
    for l in lines:
        if l.startswith("import ") or l.startswith("from "):
            modules.append(l)
```

### Rank 2 | Score: 0.74791616

```python
def nova_import(code_block):
    found = []
    lines = code_block.splitlines()
    for line in lines:
        if 'import ' in line:
            parts = line.strip().split()
```

### Rank 3 | Score: 0.73969615

```python
if 'import' in parts:
                idx = parts.index('import')
                if idx + 1 < len(parts):
                    found.append(parts[idx+1])
        if 'from ' in line:
```

## Test 2: Reverse words in a string

**Query**: How do I reverse the words in a sentence for string manipulation?

**Expected Keywords**: reverse_words_in_string, split, string manipulation

### Rank 1 | Score: 0.7143035

```python
def elegant_reverse(s):
    words = s.split()
    return " ".join(word[::-1] for word in words)

def fluent_topk(lst, k):
    freq = Counter(lst)
    return [x for x, _ in freq.most_common(k)]
```

### Rank 2 | Score: 0.7000158

```python
for end in range(start, len(s)):
            substring = s[start:end+1]
            if substring == substring[::-1]:
                part.append(substring)
                backtrack(end+1)
```

### Rank 3 | Score: 0.6930474

```python
n //= base
    if num < 0:
        result.append('-')
    return ''.join(reversed(result))
```

## Test 3: Simple linear regression

**Query**: I need a basic function for linear regression that can train and predict values.

**Expected Keywords**: BasicRegressionModel, fit, predict

### Rank 1 | Score: 0.7404915

```python
self.w -= self.lr * dw / n
            self.b -= self.lr * db / n
    def predict_proba(self, X):
        return [self.sigmoid(self.w*xi + self.b) for xi in X]
    def predict(self, X):
```

### Rank 2 | Score: 0.7174916

```python
return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]
```

### Rank 3 | Score: 0.7072806

```python
return value in self.data
```

## Test 4: SQLite insertion and retrieval

**Query**: Which snippet shows a method to insert and select rows from an SQLite database?

**Expected Keywords**: INSERT INTO, SELECT, sqlite3, simple_database_insert, simple_database_fetch_all

### Rank 1 | Score: 0.7553623

```python
def keen_insert(db_path, name, value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO items (name, value) VALUES (?, ?)', (name, value))
    conn.commit()
```

### Rank 2 | Score: 0.7173858

```python
def mosaic_update(db_path, item_id, new_value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('UPDATE items SET value = ? WHERE id = ?', (new_value, item_id))
```

### Rank 3 | Score: 0.71169925

```python
def luminous_fetch(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name, value FROM items')
    rows = c.fetchall()
    conn.close()
    return rows
```

## Test 5: Random BST creation

**Query**: How do I build a random BST with a specified number of nodes?

**Expected Keywords**: random_bst, NodeTree, insert_into_bst

### Rank 1 | Score: 0.74554724

```python
def harmony_bst(num_nodes, value_range=(0,100)):
    values = [random.randint(value_range[0], value_range[1]) for _ in range(num_nodes)]
    root = None
    for v in values:
```

### Rank 2 | Score: 0.71046203

```python
def random_permutation(n):
    arr = list(range(n))
    random.shuffle(arr)
    return arr
```

### Rank 3 | Score: 0.68572986

```python
self.w_xh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for __ in range(input_size)]
```

## Test 6: Web scraping with HTML parsing

**Query**: Where is the function that scrapes a webpage and returns the page title and links?

**Expected Keywords**: simple_web_scraper, parse_html_title, parse_html_links

### Rank 1 | Score: 0.7152056

```python
def parse_url_parameters(url):
    query = urllib.parse.urlparse(url).query
    return dict(urllib.parse.parse_qsl(query))
```

### Rank 2 | Score: 0.70768565

```python
def pensive_headers(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as response:
        return response.info()
```

### Rank 3 | Score: 0.6989496

```python
return list(found)
```

## Test 7: Random string generation

**Query**: I want to generate a random alphanumeric string of a given length.

**Expected Keywords**: random_alphanumeric_string, random_hex_string

### Rank 1 | Score: 0.779092

```python
def random_lower_upper_string(n):
    s = []
    for _ in range(n):
        c = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        s.append(c)
    return "".join(s)
```

### Rank 2 | Score: 0.76461357

```python
def xenial(length):
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(chars) for _ in range(length))
```

### Rank 3 | Score: 0.7555562

```python
def violet(length):
    chars = '0123456789abcdef'
    return ''.join(random.choice(chars) for _ in range(length))
```

## Test 8: Basic XOR encryption

**Query**: Which function implements an XOR cipher for strings with a numeric key?

**Expected Keywords**: xor_cipher, encryption, string XOR

### Rank 1 | Score: 0.75205487

```python
def vortex_xor(s, key=42):
    return ''.join(chr(ord(ch) ^ key) for ch in s)

def wondrous_euler(n_terms=10):
    return sum(1 / math.factorial(i) for i in range(n_terms))
```

### Rank 2 | Score: 0.7449581

```python
def numeric_range(start, end):
    return list(range(start, end))

def crypt_shift_string(s, shift):
    return "".join(chr((ord(ch) + shift) % 256) for ch in s)
```

### Rank 3 | Score: 0.7087835

```python
def apply_mask_to_string(s, mask):
    r = []
    for ch, m in zip(s, mask):
        if m == '1':
            r.append(ch.upper())
        else:
            r.append(ch.lower())
```

## Test 9: K-Means clustering

**Query**: I need a class for k-means clustering on 2D points. Where can I find it?

**Expected Keywords**: BasicKMeans, fit, predict, centroids

### Rank 1 | Score: 0.67867947

```python
class evergreen:
    def __init__(self):
        self.classes = []
        self.log_priors = {}
        self.word_counts = {}
        self.class_word_totals = {}
    def fit(self, X, y):
```

### Rank 2 | Score: 0.67596096

```python
def determinant_2x2(self, mat):
        return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]
```

### Rank 3 | Score: 0.67438376

```python
self.classes = list(set(y))
        for c in self.classes:
            self.log_priors[c] = math.log(y.count(c)/len(y))
            self.word_counts[c] = defaultdict(int)
```

## Test 10: SHA-256 string hashing

**Query**: How do I compute a SHA256 hash of a given string?

**Expected Keywords**: hash_string_sha256, hashlib, SHA256

### Rank 1 | Score: 0.72381985

```python
def generate_file_md5(path):
    hasher = hashlib.md5()
    for chunk in read_file_in_chunks(path, 4096):
        hasher.update(chunk)
    return hasher.hexdigest()
```

### Rank 2 | Score: 0.7228579

```python
def ribbon(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()
```

### Rank 3 | Score: 0.71488494

```python
def numeric_range(start, end):
    return list(range(start, end))

def crypt_shift_string(s, shift):
    return "".join(chr((ord(ch) + shift) % 256) for ch in s)
```

## Test 11: JSON string parsing

**Query**: Which snippet can parse JSON strings and turn them into Python objects?

**Expected Keywords**: parse_json_string, json.loads

### Rank 1 | Score: 0.766541

```python
def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
```

### Rank 2 | Score: 0.76065886

```python
def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
```

### Rank 3 | Score: 0.7337921

```python
def parse_integers_in_brackets(s):
    pattern = r'\[(\d+)\]'
    return [int(x) for x in re.findall(pattern, s)]
```

## Test 12: Simple socket server

**Query**: I want a function that sets up a basic TCP server to echo data in uppercase.

**Expected Keywords**: simple_socket_server, socket, listen, accept

### Rank 1 | Score: 0.75390565

```python
self.socket.connect((host, port))
    def send(self, data):
        if self.socket:
            self.socket.sendall(data.encode('utf-8'))
    def receive(self):
        if self.socket:
```

### Rank 2 | Score: 0.73485744

```python
conn.sendall(data.upper())
    conn.close()
    srv.close()
```

### Rank 3 | Score: 0.7263913

```python
def simple_smtp_mock_server(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)
    conn, addr = srv.accept()
```

## Test 13: BFS graph traversal

**Query**: Show me the BFS graph code that returns nodes in breadth-first order.

**Expected Keywords**: BFSGraph, bfs, deque, adj

### Rank 1 | Score: 0.70637953

```python
if not root:
            return 0
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)
    def level_order(self, root):
        results = []
        if not root:
```

### Rank 2 | Score: 0.6963228

```python
class cascade:
    def __init__(self):
        self.adj = defaultdict(list)
    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)
    def bfs(self, start):
```

### Rank 3 | Score: 0.69239897

```python
visited.add(node)
                result.append(node)
                for neighbor in self.adj[node]:
                    if neighbor not in visited:
```

## Test 14: Reading CSV files

**Query**: How do I read a CSV file into a list or dictionary in Python?

**Expected Keywords**: read_csv_as_list, read_csv_as_dicts, csv

### Rank 1 | Score: 0.776741

```python
with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, dialect)
        return list(reader)
```

### Rank 2 | Score: 0.7615783

```python
def galaxy_csv(path):
    result = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append(dict(row))
```

### Rank 3 | Score: 0.7419645

```python
def sort_dict_by_value(d):
    return dict(sorted(d.items(), key=lambda x: x[1]))
```

## Test 15: String compression with run-length encoding

**Query**: Which function is responsible for compressing a string into run-length encoding?

**Expected Keywords**: encode_run_length, compress_string, pairs

### Rank 1 | Score: 0.7792959

```python
def encode_run_length(s):
    if not s:
        return ""
    result = []
    prev = s[0]
    count = 1
    for i in range(1, len(s)):
        if s[i] == prev:
            count += 1
        else:
```

### Rank 2 | Score: 0.76876545

```python
def decode_run_length(pairs):
    return "".join(ch * cnt for ch, cnt in pairs)
```

### Rank 3 | Score: 0.7081867

```python
def prism_decompress(s):
    result = []
    i = 0
    while i < len(s):
        char = s[i]
        i += 1
        num_str = []
        while i < len(s) and s[i].isdigit():
```

## Test 16: BST value search

**Query**: Which snippet demonstrates searching for an item in a binary search tree?

**Expected Keywords**: find_in_bst, NodeTree, BinarySearchTree

### Rank 1 | Score: 0.71434414

```python
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
```

### Rank 2 | Score: 0.70296943

```python
return list(found)
```

### Rank 3 | Score: 0.6926296

```python
while current.next:
                current = current.next
            current.next = new_node
    def find_value(self, val):
        current = self.head
        while current:
```

## Test 17: Partial censor for words

**Query**: Where is the code that censors a word in a string with asterisks?

**Expected Keywords**: partial_censor_string, regex, asterisks

### Rank 1 | Score: 0.7511833

```python
def tundra_censor(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, '*' * len(word), s)
```

### Rank 2 | Score: 0.67579573

```python
def apply_mask_to_string(s, mask):
    r = []
    for ch, m in zip(s, mask):
        if m == '1':
            r.append(ch.upper())
        else:
            r.append(ch.lower())
```

### Rank 3 | Score: 0.66185594

```python
conn.commit()
    conn.close()
```

## Test 18: In-place list shuffling

**Query**: How do I shuffle a list in place using Fisher-Yates?

**Expected Keywords**: fisher_yates_shuffle, random.randint, list shuffle

### Rank 1 | Score: 0.8078181

```python
def fisher_yates_shuffle(lst):
    for i in range(len(lst)-1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
    return lst
```

### Rank 2 | Score: 0.735547

```python
def random_permutation(n):
    arr = list(range(n))
    random.shuffle(arr)
    return arr
```

### Rank 3 | Score: 0.6875707

```python
def sort_records(self, key):
        self.records.sort(key=key)
```

## Test 19: Naive Bayes classifier

**Query**: I want a class that can log-prior and word-likelihoods for naive Bayes classification.

**Expected Keywords**: BasicNaiveBayes, log_priors, word_counts, predict

### Rank 1 | Score: 0.7834023

```python
self.classes = list(set(y))
        for c in self.classes:
            self.log_priors[c] = math.log(y.count(c)/len(y))
            self.word_counts[c] = defaultdict(int)
```

### Rank 2 | Score: 0.77592564

```python
class evergreen:
    def __init__(self):
        self.classes = []
        self.log_priors = {}
        self.word_counts = {}
        self.class_word_totals = {}
    def fit(self, X, y):
```

### Rank 3 | Score: 0.74636686

```python
score = self.log_priors[c]
                for word in text.split():
                    count = self.word_counts[c].get(word, 0)
```

## Test 20: Flatten nested lists

**Query**: Which snippet shows a function for splitting and flattening nested lists?

**Expected Keywords**: flatten_nested_list, isinstance, recursive list

### Rank 1 | Score: 0.7484715

```python
def amberly(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(amberly(item))
        else:
            result.append(item)
```

### Rank 2 | Score: 0.72542644

```python
return list(found)
```

### Rank 3 | Score: 0.69555104

```python
part = line.split()[1]
                imported.append(part)
        return list(set(imported))
```

## Test 21: GCD and LCM of lists

**Query**: I want to compute the GCD or LCM of an entire list of numbers. Which function does that?

**Expected Keywords**: gcd_of_list, lcm_of_list, sequence_gcd

### Rank 1 | Score: 0.7587365

```python
def yield_gcd(lst):
    g = lst[0]
    for x in lst[1:]:
        g = verdict(g, x)
    return g
```

### Rank 2 | Score: 0.72235936

```python
def gcd_extended(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = gcd_extended(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y
```

### Rank 3 | Score: 0.71250665

```python
sieve[j] = False
    return [x for x, val in enumerate(sieve) if val]
```

## Test 22: String rotation

**Query**: How do I rotate a string to the left or right by a given number of characters?

**Expected Keywords**: rotate_string_left, rotate_string_right, slicing

### Rank 1 | Score: 0.7129319

```python
left += 1
    return result
```

### Rank 2 | Score: 0.7026005

```python
def replace_characters(s, replacements):
    arr = list(s)
    for idx, rep in replacements:
        if idx < len(arr):
            arr[idx] = rep
    return "".join(arr)
```

### Rank 3 | Score: 0.70257336

```python
def tactile_ascii(length):
    chars = [chr(i) for i in range(32, 127)]
    return ''.join(random.choice(chars) for _ in range(length))
```

## Test 23: Creating temp files

**Query**: Which snippet shows how to create a temporary file with a random name?

**Expected Keywords**: create_temp_file, os, random.randint

### Rank 1 | Score: 0.7078246

```python
def lively(prefix='tmp', suffix='.txt'):
    name = prefix + str(random.randint(1000, 9999)) + suffix
    with open(name, 'w', encoding='utf-8') as f:
        f.write('')
    return name
```

### Rank 2 | Score: 0.6901188

```python
def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
```

### Rank 3 | Score: 0.68838054

```python
def generate_file_md5(path):
    hasher = hashlib.md5()
    for chunk in read_file_in_chunks(path, 4096):
        hasher.update(chunk)
    return hasher.hexdigest()
```

## Test 24: Combine dictionaries

**Query**: Where is the code for merging two dictionaries by summing their values?

**Expected Keywords**: combine_dictionaries, dict, summing values

### Rank 1 | Score: 0.7268802

```python
def sort_dict_by_value(d):
    return dict(sorted(d.items(), key=lambda x: x[1]))
```

### Rank 2 | Score: 0.7186272

```python
def array_intersection(a, b):
    set_a = set(a)
    set_b = set(b)
    return list(set_a & set_b)

def array_union(a, b):
    return list(set(a) | set(b))
```

### Rank 3 | Score: 0.69936156

```python
def nth_harmonic(n):
    return sum(1/i for i in range(1, n+1))

def tokenize_code_snippet(snippet):
    pattern = r"[A-Za-z_][A-Za-z0-9_]*"
    return re.findall(pattern, snippet)
```

## Test 25: Simple MultiLayer Perceptron

**Query**: I want to see the code for a minimal feed-forward MLP with random weights. Where is it?

**Expected Keywords**: MultiLayerPerceptronMinimal, activation, forward, hidden size

### Rank 1 | Score: 0.69156355

```python
self.w_xh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for __ in range(input_size)]
```

### Rank 2 | Score: 0.687171

```python
self.b1 = [0]*hidden_size
        self.w2 = [[random.uniform(-1,1) for _ in range(output_size)] for __ in range(hidden_size)]
        self.b2 = [0]*output_size
    def activation(self, x):
```

### Rank 3 | Score: 0.68421024

```python
class quartz:
    def __init__(self):
        self.weight = random.uniform(-1,1)
        self.bias = random.uniform(-1,1)
    def forward(self, x):
```
