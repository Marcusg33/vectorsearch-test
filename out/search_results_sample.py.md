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

### Rank 2 | Score: 0.7677087

```python
class ImportResolver2:
    def find_imports(self, code_str):
        pattern = r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)|import\s+([a-zA-Z0-9_\.]+))'
```

### Rank 3 | Score: 0.76259434

```python
for line in lines:
            if line.strip().startswith('import '):
                part = line.strip().split('import ')[1].split()[0]
                imported_modules.append(part)
```

At least one result contains the expected keywords. ✅

## Test 2: Reverse words in a string

**Query**: How do I reverse the words in a sentence for string manipulation?

**Expected Keywords**: reverse_words_in_string, split, string manipulation

### Rank 1 | Score: 0.83365464

```python
def reverse_words_in_string(s):
    words = s.split()
    return " ".join(words[::-1])
```

### Rank 2 | Score: 0.75117666

```python
def rotate_string_left(s, n):
    n = n % len(s)
    return s[n:] + s[:n]

def rotate_string_right(s, n):
    n = n % len(s)
    return s[-n:] + s[:-n]

def reverse_string(s):
    return s[::-1]
```

### Rank 3 | Score: 0.7426961

```python
def reverse_each_word(s):
    words = s.split()
    return " ".join(word[::-1] for word in words)

def top_k_frequent(lst, k):
    freq = Counter(lst)
    return [x for x, _ in freq.most_common(k)]
```

No result contained all expected keywords. ❌

## Test 3: Simple linear regression

**Query**: I need a basic function for linear regression that can train and predict values.

**Expected Keywords**: BasicRegressionModel, fit, predict

### Rank 1 | Score: 0.7541709

```python
self.bias -= self.lr * db
    def predict(self, X):
        return [self.weight * x + self.bias for x in X]
```

### Rank 2 | Score: 0.7404915

```python
self.w -= self.lr * dw / n
            self.b -= self.lr * db / n
    def predict_proba(self, X):
        return [self.sigmoid(self.w*xi + self.b) for xi in X]
    def predict(self, X):
```

### Rank 3 | Score: 0.7318808

```python
class BasicRegressionModel:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iters = iterations
        self.weight = 0.0
```

No result contained all expected keywords. ❌

## Test 4: SQLite insertion and retrieval

**Query**: Which snippet shows a method to insert and select rows from an SQLite database?

**Expected Keywords**: INSERT INTO, SELECT, sqlite3, simple_database_insert, simple_database_fetch_all

### Rank 1 | Score: 0.77502173

```python
def simple_database_insert(db_path, name, value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO items (name, value) VALUES (?, ?)', (name, value))
```

### Rank 2 | Score: 0.74645555

```python
def simple_database_fetch_all(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name, value FROM items')
    rows = c.fetchall()
    conn.close()
```

### Rank 3 | Score: 0.74480397

```python
def simple_database_init(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, value REAL)')
```

No result contained all expected keywords. ❌

## Test 5: Random BST creation

**Query**: How do I build a random BST with a specified number of nodes?

**Expected Keywords**: random_bst, NodeTree, insert_into_bst

### Rank 1 | Score: 0.7971541

```python
def random_bst(num_nodes, value_range=(0,100)):
    values = [random.randint(value_range[0], value_range[1]) for _ in range(num_nodes)]
    root = None
    for v in values:
```

### Rank 2 | Score: 0.73401743

```python
def generate_random_seeded_numbers(seed_val, n, start=0, end=100):
    r = random.Random(seed_val)
    return [r.randint(start, end) for _ in range(n)]
```

### Rank 3 | Score: 0.7220427

```python
def build_adjacency_matrix(edges, n_nodes):
    matrix = [[0]*n_nodes for _ in range(n_nodes)]
    for (u, v) in edges:
        matrix[u][v] = 1
        matrix[v][u] = 1
    return matrix
```

No result contained all expected keywords. ❌

## Test 6: Web scraping with HTML parsing

**Query**: Where is the function that scrapes a webpage and returns the page title and links?

**Expected Keywords**: simple_web_scraper, parse_html_title, parse_html_links

### Rank 1 | Score: 0.8088495

```python
def simple_web_scraper(url):
    try:
        data = small_web_request(url)
        title = parse_html_title(data)
        links = parse_html_links(data)
        return title, links
    except:
```

### Rank 2 | Score: 0.7678715

```python
def parse_html_title(html):
    pattern = r'<title>(.*?)</title>'
    matches = re.findall(pattern, html, re.IGNORECASE)
    if matches:
        return matches[0]
    return None
```

### Rank 3 | Score: 0.7603431

```python
def parse_html_links(html):
    pattern = r'href=["\'](.*?)["\']'
    return re.findall(pattern, html)
```

At least one result contains the expected keywords. ✅

## Test 7: Random string generation

**Query**: I want to generate a random alphanumeric string of a given length.

**Expected Keywords**: random_alphanumeric_string, random_hex_string

### Rank 1 | Score: 0.8606444

```python
def random_alphanumeric_string(length):
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(chars) for _ in range(length))
```

### Rank 2 | Score: 0.8119831

```python
def random_hex_string(length):
    chars = '0123456789abcdef'
    return ''.join(random.choice(chars) for _ in range(length))
```

### Rank 3 | Score: 0.8113759

```python
def random_ascii_string(length):
    chars = [chr(i) for i in range(32, 127)]
    return ''.join(random.choice(chars) for _ in range(length))
```

No result contained all expected keywords. ❌

## Test 8: Basic XOR encryption

**Query**: Which function implements an XOR cipher for strings with a numeric key?

**Expected Keywords**: xor_cipher, encryption, string XOR

### Rank 1 | Score: 0.8133589

```python
def xor_cipher(s, key=42):
    return ''.join(chr(ord(ch) ^ key) for ch in s)

def approximate_euler(n_terms=10):
    return sum(1 / math.factorial(i) for i in range(n_terms))
```

### Rank 2 | Score: 0.74509585

```python
def numeric_range(start, end):
    return list(range(start, end))

def crypt_shift_string(s, shift):
    return "".join(chr((ord(ch) + shift) % 256) for ch in s)
```

### Rank 3 | Score: 0.73029894

```python
def decompress_string(s):
    result = []
    i = 0
    while i < len(s):
        char = s[i]
        i += 1
        num_str = []
        while i < len(s) and s[i].isdigit():
```

No result contained all expected keywords. ❌

## Test 9: K-Means clustering

**Query**: I need a class for k-means clustering on 2D points. Where can I find it?

**Expected Keywords**: BasicKMeans, fit, predict, centroids

### Rank 1 | Score: 0.74507123

```python
dists = [math.dist(point, c) for c in self.centroids]
                cluster_idx = dists.index(min(dists))
                clusters[cluster_idx].append(point)
```

### Rank 2 | Score: 0.74459374

```python
self.centroids = random.sample(data, self.k)
        for _ in range(self.iters):
            clusters = [[] for __ in range(self.k)]
            for point in data:
```

### Rank 3 | Score: 0.7337402

```python
class BasicKMeans:
    def __init__(self, k=2, iterations=100):
        self.k = k
        self.iters = iterations
        self.centroids = []
    def fit(self, data):
```

No result contained all expected keywords. ❌

## Test 10: SHA-256 string hashing

**Query**: How do I compute a SHA256 hash of a given string?

**Expected Keywords**: hash_string_sha256, hashlib, SHA256

### Rank 1 | Score: 0.8063562

```python
def hash_string_sha256(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()
```

### Rank 2 | Score: 0.72381985

```python
def generate_file_md5(path):
    hasher = hashlib.md5()
    for chunk in read_file_in_chunks(path, 4096):
        hasher.update(chunk)
    return hasher.hexdigest()
```

### Rank 3 | Score: 0.72038406

```python
def random_hex_string(length):
    chars = '0123456789abcdef'
    return ''.join(random.choice(chars) for _ in range(length))
```

At least one result contains the expected keywords. ✅

## Test 11: JSON string parsing

**Query**: Which snippet can parse JSON strings and turn them into Python objects?

**Expected Keywords**: parse_json_string, json.loads

### Rank 1 | Score: 0.815004

```python
def create_json_string(obj):
    return json.dumps(obj)
```

### Rank 2 | Score: 0.7757572

```python
def small_web_request(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8')

def parse_json_string(json_str):
    return json.loads(json_str)
```

### Rank 3 | Score: 0.76621145

```python
def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
```

At least one result contains the expected keywords. ✅

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

### Rank 2 | Score: 0.7436855

```python
def simple_socket_server(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)
    conn, addr = srv.accept()
```

### Rank 3 | Score: 0.73989093

```python
def random_uppercase_letters(n):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choice(chars) for _ in range(n))
```

At least one result contains the expected keywords. ✅

## Test 13: BFS graph traversal

**Query**: Show me the BFS graph code that returns nodes in breadth-first order.

**Expected Keywords**: BFSGraph, bfs, deque, adj

### Rank 1 | Score: 0.74336225

```python
class BFSGraph:
    def __init__(self):
        self.adj = defaultdict(list)
    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)
    def bfs(self, start):
```

### Rank 2 | Score: 0.72144043

```python
def build_adjacency_matrix(edges, n_nodes):
    matrix = [[0]*n_nodes for _ in range(n_nodes)]
    for (u, v) in edges:
        matrix[u][v] = 1
        matrix[v][u] = 1
    return matrix
```

### Rank 3 | Score: 0.70640165

```python
if not root:
            return 0
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)
    def level_order(self, root):
        results = []
        if not root:
```

No result contained all expected keywords. ❌

## Test 14: Reading CSV files

**Query**: How do I read a CSV file into a list or dictionary in Python?

**Expected Keywords**: read_csv_as_list, read_csv_as_dicts, csv

### Rank 1 | Score: 0.8416888

```python
def read_csv_as_dicts(path):
    result = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
```

### Rank 2 | Score: 0.83112293

```python
def read_csv_as_list(path):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
```

### Rank 3 | Score: 0.799479

```python
def write_list_as_csv(path, data):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)
```

No result contained all expected keywords. ❌

## Test 15: String compression with run-length encoding

**Query**: Which function is responsible for compressing a string into run-length encoding?

**Expected Keywords**: encode_run_length, compress_string, pairs

### Rank 1 | Score: 0.78283864

```python
def compress_string(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(len(s)-1):
        if s[i] == s[i+1]:
            count += 1
        else:
```

### Rank 2 | Score: 0.7792959

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

### Rank 3 | Score: 0.76876545

```python
def decode_run_length(pairs):
    return "".join(ch * cnt for ch, cnt in pairs)
```

No result contained all expected keywords. ❌

## Test 16: BST value search

**Query**: Which snippet demonstrates searching for an item in a binary search tree?

**Expected Keywords**: find_in_bst, NodeTree, BinarySearchTree

### Rank 1 | Score: 0.7638031

```python
class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, val):
        if self.root is None:
            self.root = NodeTree(val)
        else:
```

### Rank 2 | Score: 0.72781587

```python
def binary_search(lst, val):
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] == val:
            return mid
        elif lst[mid] < val:
```

### Rank 3 | Score: 0.71940315

```python
def find_in_bst(root, val):
    current = root
    while current:
        if current.val == val:
            return True
        elif val < current.val:
            current = current.left
```

No result contained all expected keywords. ❌

## Test 17: Partial censor for words

**Query**: Where is the code that censors a word in a string with asterisks?

**Expected Keywords**: partial_censor_string, regex, asterisks

### Rank 1 | Score: 0.7664888

```python
def partial_censor_string(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, '*' * len(word), s)
```

### Rank 2 | Score: 0.69237196

```python
remove_directory(p)
```

### Rank 3 | Score: 0.68302095

```python
def to_upper_case(s):
    return s.upper()

def to_lower_case(s):
    return s.lower()

def strip_spaces(s):
    return s.strip()
```

No result contained all expected keywords. ❌

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

### Rank 2 | Score: 0.7359755

```python
def random_choice_from_list(lst):
    if not lst:
        return None
    return random.choice(lst)

def shuffle_list(lst):
    random.shuffle(lst)
    return lst
```

### Rank 3 | Score: 0.73566455

```python
def random_permutation(n):
    arr = list(range(n))
    random.shuffle(arr)
    return arr
```

No result contained all expected keywords. ❌

## Test 19: Naive Bayes classifier

**Query**: I want a class that can log-prior and word-likelihoods for naive Bayes classification.

**Expected Keywords**: BasicNaiveBayes, log_priors, word_counts, predict

### Rank 1 | Score: 0.7843742

```python
class BasicNaiveBayes:
    def __init__(self):
        self.classes = []
        self.log_priors = {}
        self.word_counts = {}
        self.class_word_totals = {}
    def fit(self, X, y):
```

### Rank 2 | Score: 0.7834023

```python
self.classes = list(set(y))
        for c in self.classes:
            self.log_priors[c] = math.log(y.count(c)/len(y))
            self.word_counts[c] = defaultdict(int)
```

### Rank 3 | Score: 0.74636686

```python
score = self.log_priors[c]
                for word in text.split():
                    count = self.word_counts[c].get(word, 0)
```

No result contained all expected keywords. ❌

## Test 20: Flatten nested lists

**Query**: Which snippet shows a function for splitting and flattening nested lists?

**Expected Keywords**: flatten_nested_list, isinstance, recursive list

### Rank 1 | Score: 0.8160844

```python
def flatten_nested_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested_list(item))
        else:
```

### Rank 2 | Score: 0.73390234

```python
return list(duplicates)
```

### Rank 3 | Score: 0.73006

```python
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
```

No result contained all expected keywords. ❌

## Test 21: GCD and LCM of lists

**Query**: I want to compute the GCD or LCM of an entire list of numbers. Which function does that?

**Expected Keywords**: gcd_of_list, lcm_of_list, sequence_gcd

### Rank 1 | Score: 0.7886857

```python
def gcd_of_list(lst):
    g = lst[0]
    for x in lst[1:]:
        g = sequence_gcd(g, x)
    return g
```

### Rank 2 | Score: 0.7714827

```python
def lcm_of_list(lst):
    def lcm(a, b):
        return abs(a*b) // sequence_gcd(a, b) if a and b else 0
    current = lst[0]
    for x in lst[1:]:
        current = lcm(current, x)
```

### Rank 3 | Score: 0.73951066

```python
def sequence_gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
```

No result contained all expected keywords. ❌

## Test 22: String rotation

**Query**: How do I rotate a string to the left or right by a given number of characters?

**Expected Keywords**: rotate_string_left, rotate_string_right, slicing

### Rank 1 | Score: 0.8297801

```python
def rotate_string_left(s, n):
    n = n % len(s)
    return s[n:] + s[:n]

def rotate_string_right(s, n):
    n = n % len(s)
    return s[-n:] + s[:-n]

def reverse_string(s):
    return s[::-1]
```

### Rank 2 | Score: 0.7400098

```python
def chunk_string(s, size):
    return [s[i:i+size] for i in range(0, len(s), size)]

def is_string_rotation(s1, s2):
    if len(s1) != len(s2):
        return False
    return s2 in (s1+s1)
```

### Rank 3 | Score: 0.7269967

```python
def random_ascii_string(length):
    chars = [chr(i) for i in range(32, 127)]
    return ''.join(random.choice(chars) for _ in range(length))
```

No result contained all expected keywords. ❌

## Test 23: Creating temp files

**Query**: Which snippet shows how to create a temporary file with a random name?

**Expected Keywords**: create_temp_file, os, random.randint

### Rank 1 | Score: 0.79314315

```python
def create_temp_file(prefix='tmp', suffix='.txt'):
    name = prefix + str(random.randint(1000, 9999)) + suffix
    with open(name, 'w', encoding='utf-8') as f:
        f.write('')
    return name
```

### Rank 2 | Score: 0.7120798

```python
def create_json_string(obj):
    return json.dumps(obj)
```

### Rank 3 | Score: 0.6940055

```python
def list_directory_contents(path):
    return os.listdir(path)

def create_directory(path):
    os.makedirs(path, exist_ok=True)
```

No result contained all expected keywords. ❌

## Test 24: Combine dictionaries

**Query**: Where is the code for merging two dictionaries by summing their values?

**Expected Keywords**: combine_dictionaries, dict, summing values

### Rank 1 | Score: 0.79601276

```python
def combine_dictionaries(dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            result[k] = result.get(k, 0) + v
    return result
```

### Rank 2 | Score: 0.72793067

```python
def use_ordered_dict(pairs):
    d = OrderedDict()
    for k, v in pairs:
        d[k] = v
    return d
```

### Rank 3 | Score: 0.7268802

```python
def sort_dict_by_value(d):
    return dict(sorted(d.items(), key=lambda x: x[1]))
```

No result contained all expected keywords. ❌

## Test 25: Simple MultiLayer Perceptron

**Query**: I want to see the code for a minimal feed-forward MLP with random weights. Where is it?

**Expected Keywords**: MultiLayerPerceptronMinimal, activation, forward, hidden size

### Rank 1 | Score: 0.7326298

```python
class SigmoidNeuron:
    def __init__(self):
        self.weight = random.uniform(-1,1)
        self.bias = random.uniform(-1,1)
    def forward(self, x):
```

### Rank 2 | Score: 0.72890586

```python
class MultiLayerPerceptronMinimal:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = [[random.uniform(-1,1) for _ in range(hidden_size)] for __ in range(input_size)]
```

### Rank 3 | Score: 0.6940125

```python
self.bias -= self.lr * db
    def predict(self, X):
        return [self.weight * x + self.bias for x in X]
```

No result contained all expected keywords. ❌
