import sys
import os
import math
import random
import sqlite3
import urllib.request
import urllib.parse
import http.client
import socket
import itertools
import json
import re
import csv
import hashlib
from collections import defaultdict, deque, OrderedDict, Counter
from functools import lru_cache
from datetime import datetime
from decimal import Decimal

def simple_adder(a, b):
    return a + b

def simple_subtractor(a, b):
    return a - b

def simple_multiplier(a, b):
    return a * b

def simple_divider(a, b):
    if b == 0:
        return None
    return a / b

def sqrt_approx(x):
    return x ** 0.5

def exponent_power(base, exp):
    return base ** exp

def custom_abs(x):
    if x < 0:
        return -x
    return x

def int_to_binary_string(x):
    return bin(x)[2:]

def binary_string_to_int(s):
    return int(s, 2)

def rotate_string_left(s, n):
    n = n % len(s)
    return s[n:] + s[:n]

def rotate_string_right(s, n):
    n = n % len(s)
    return s[-n:] + s[:-n]

def reverse_string(s):
    return s[::-1]

def to_upper_case(s):
    return s.upper()

def to_lower_case(s):
    return s.lower()

def strip_spaces(s):
    return s.strip()

def random_choice_from_list(lst):
    if not lst:
        return None
    return random.choice(lst)

def shuffle_list(lst):
    random.shuffle(lst)
    return lst

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def generate_random_integers(n, start=0, end=100):
    result = []
    for _ in range(n):
        result.append(random.randint(start, end))
    return result

def create_counter_dictionary(lst):
    d = {}
    for item in lst:
        if item in d:
            d[item] += 1
        else:
            d[item] = 1
    return d

def set_operations_union(s1, s2):
    return s1 | s2

def set_operations_intersection(s1, s2):
    return s1 & s2

def set_operations_difference(s1, s2):
    return s1 - s2

class BasicCounter:
    def __init__(self):
        self.value = 0
    def increment(self):
        self.value += 1
    def decrement(self):
        self.value -= 1
    def reset(self):
        self.value = 0
    def get_value(self):
        return self.value

class SimpleQueue:
    def __init__(self):
        self.data = deque()
    def enqueue(self, item):
        self.data.append(item)
    def dequeue(self):
        if self.data:
            return self.data.popleft()
        return None
    def is_empty(self):
        return len(self.data) == 0
    def size(self):
        return len(self.data)

class SimpleStack:
    def __init__(self):
        self.data = []
    def push(self, item):
        self.data.append(item)
    def pop(self):
        if self.data:
            return self.data.pop()
        return None
    def is_empty(self):
        return len(self.data) == 0
    def size(self):
        return len(self.data)

class DoublyLinkedListNode:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def append(self, value):
        new_node = DoublyLinkedListNode(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
    def prepend(self, value):
        new_node = DoublyLinkedListNode(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
    def remove(self, value):
        current = self.head
        while current:
            if current.value == value:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                return
            current = current.next

def quicksort(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[0]
    less = [x for x in lst[1:] if x <= pivot]
    greater = [x for x in lst[1:] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)

def mergesort(lst):
    if len(lst) > 1:
        mid = len(lst) // 2
        left_half = lst[:mid]
        right_half = lst[mid:]
        mergesort(left_half)
        mergesort(right_half)
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                lst[k] = left_half[i]
                i += 1
            else:
                lst[k] = right_half[j]
                j += 1
            k += 1
        while i < len(left_half):
            lst[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            lst[k] = right_half[j]
            j += 1
            k += 1
    return lst

def factorial_iterative(n):
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def factorial_recursive(n):
    if n < 0:
        return None
    if n == 0:
        return 1
    return n * factorial_recursive(n - 1)

def fib_iterative(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq

def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def parse_int_list_from_string(s):
    parts = s.split()
    return [int(p) for p in parts if p.isdigit()]

def join_numbers_with_comma(lst):
    return ",".join(str(x) for x in lst)

def reverse_words_in_string(s):
    words = s.split()
    return " ".join(words[::-1])

def random_hex_string(length):
    chars = '0123456789abcdef'
    return ''.join(random.choice(chars) for _ in range(length))

def random_alphanumeric_string(length):
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(chars) for _ in range(length))

def count_vowels(s):
    c = 0
    vowels = set('aeiouAEIOU')
    for ch in s:
        if ch in vowels:
            c += 1
    return c

def list_combinations(lst, r):
    return list(itertools.combinations(lst, r))

def flatten_nested_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested_list(item))
        else:
            result.append(item)
    return result

def read_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def check_file_exists(path):
    return os.path.isfile(path)

def simple_file_copy(src, dst):
    with open(src, 'rb') as sf:
        with open(dst, 'wb') as df:
            df.write(sf.read())

def file_line_count(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def current_working_directory():
    return os.getcwd()

def list_directory_contents(path):
    return os.listdir(path)

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)

def remove_directory(path):
    if os.path.exists(path):
        os.rmdir(path)

def create_temp_file(prefix='tmp', suffix='.txt'):
    name = prefix + str(random.randint(1000, 9999)) + suffix
    with open(name, 'w', encoding='utf-8') as f:
        f.write('')
    return name

class StringManipulator:
    def __init__(self, text):
        self.text = text
    def replace_substring(self, old, new):
        self.text = self.text.replace(old, new)
    def get_text(self):
        return self.text
    def split_words(self):
        return self.text.split()
    def join_with_dash(self):
        return "-".join(self.split_words())

class ImportResolver1:
    def __init__(self):
        pass
    def resolve_imports_from_code_string(self, code_str):
        lines = code_str.split('\n')
        imported_modules = []
        for line in lines:
            if line.strip().startswith('import '):
                part = line.strip().split('import ')[1].split()[0]
                imported_modules.append(part)
            elif line.strip().startswith('from '):
                part = line.strip().split()[1]
                imported_modules.append(part)
        return list(set(imported_modules))

class ImportResolver2:
    def find_imports(self, code_str):
        pattern = r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)|import\s+([a-zA-Z0-9_\.]+))'
        matches = re.findall(pattern, code_str, re.MULTILINE)
        result = set()
        for m in matches:
            if m[0]:
                result.add(m[0])
            else:
                result.add(m[1])
        return list(result)

def small_web_request(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8')

def parse_json_string(json_str):
    return json.loads(json_str)

def create_json_string(obj):
    return json.dumps(obj)

def count_occurrences_in_file(path, term):
    c = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            c += line.count(term)
    return c

def hash_string_sha256(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def simple_socket_server(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)
    conn, addr = srv.accept()
    data = conn.recv(1024)
    conn.sendall(data.upper())
    conn.close()
    srv.close()

def simple_socket_client(host, port, message):
    cl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cl.connect((host, port))
    cl.sendall(message.encode('utf-8'))
    data = cl.recv(1024)
    cl.close()
    return data.decode('utf-8')

def sequence_gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def prime_check(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def next_prime_after(n):
    candidate = n + 1
    while True:
        if prime_check(candidate):
            return candidate
        candidate += 1

def list_primes_upto(n):
    result = []
    for num in range(2, n + 1):
        if prime_check(num):
            result.append(num)
    return result

def simple_time_function():
    now = datetime.now()
    return now

def decimal_division(a, b):
    A = Decimal(str(a))
    B = Decimal(str(b))
    if B == 0:
        return None
    return A / B

def use_ordered_dict(pairs):
    d = OrderedDict()
    for k, v in pairs:
        d[k] = v
    return d

def group_words_by_length(words):
    d = defaultdict(list)
    for w in words:
        d[len(w)].append(w)
    return dict(d)

def set_env_variable(key, value):
    os.environ[key] = value

def get_env_variable(key):
    return os.environ.get(key, None)

def star_pattern1(rows):
    result = []
    for i in range(rows):
        result.append('*' * (i+1))
    return '\n'.join(result)

def star_pattern2(rows):
    result = []
    for i in range(rows, 0, -1):
        result.append('*' * i)
    return '\n'.join(result)

def bubble_sort(lst):
    n = len(lst)
    for i in range(n):
        for j in range(0, n-i-1):
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst

def insertion_sort(lst):
    for i in range(1, len(lst)):
        key = lst[i]
        j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j+1] = lst[j]
            j -= 1
        lst[j+1] = key
    return lst

def selection_sort(lst):
    for i in range(len(lst)):
        min_idx = i
        for j in range(i+1, len(lst)):
            if lst[j] < lst[min_idx]:
                min_idx = j
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
    return lst

def super_string_combination(a, b):
    return a + b + a[::-1] + b[::-1]

def all_substrings(s):
    subs = []
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            subs.append(s[i:j])
    return subs

def is_palindrome(s):
    return s == s[::-1]

def partial_url_encode(q):
    return urllib.parse.quote(q)

def partial_url_decode(q):
    return urllib.parse.unquote(q)

def get_url_hostname(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname

def get_url_path(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.path

class MultiPurposeMath:
    def __init__(self):
        pass
    def dot_product(self, v1, v2):
        return sum(x*y for x, y in zip(v1, v2))
    def matrix_add(self, m1, m2):
        result = []
        for row1, row2 in zip(m1, m2):
            row_result = []
            for x, y in zip(row1, row2):
                row_result.append(x + y)
            result.append(row_result)
        return result
    def matrix_multiply(self, m1, m2):
        rows_m1 = len(m1)
        cols_m1 = len(m1[0])
        rows_m2 = len(m2)
        cols_m2 = len(m2[0])
        if cols_m1 != rows_m2:
            return None
        result = []
        for i in range(rows_m1):
            row_result = []
            for j in range(cols_m2):
                s = 0
                for k in range(cols_m1):
                    s += m1[i][k] * m2[k][j]
                row_result.append(s)
            result.append(row_result)
        return result
    def determinant_2x2(self, mat):
        return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]

class TextLoader:
    def __init__(self, path):
        self.path = path
        self.content = None
    def load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            self.content = f.read()
    def get_content(self):
        return self.content

class TextWriter:
    def __init__(self, path):
        self.path = path
    def write_content(self, content):
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(content)

def find_pattern_in_text(pattern, text):
    return re.findall(pattern, text)

def get_file_size(path):
    return os.path.getsize(path)

def read_csv_as_list(path):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows

def write_list_as_csv(path, data):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

def read_csv_as_dicts(path):
    result = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append(dict(row))
    return result

def write_dicts_as_csv(path, fieldnames, data):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in data:
            writer.writerow(d)

def simple_database_init(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, value REAL)')
    conn.commit()
    conn.close()

def simple_database_insert(db_path, name, value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO items (name, value) VALUES (?, ?)', (name, value))
    conn.commit()
    conn.close()

def simple_database_fetch_all(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name, value FROM items')
    rows = c.fetchall()
    conn.close()
    return rows

def simple_database_update_value(db_path, item_id, new_value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('UPDATE items SET value = ? WHERE id = ?', (new_value, item_id))
    conn.commit()
    conn.close()

def simple_database_delete_item(db_path, item_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DELETE FROM items WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()

class BasicRegressionModel:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iters = iterations
        self.weight = 0.0
        self.bias = 0.0
    def fit(self, X, y):
        n = len(X)
        for _ in range(self.iters):
            y_pred = [self.weight * x + self.bias for x in X]
            dw = (-2/n) * sum(x*(yt - yp) for x, yt, yp in zip(X, y, y_pred))
            db = (-2/n) * sum(yt - yp for yt, yp in zip(y, y_pred))
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        return [self.weight * x + self.bias for x in X]

class BasicKMeans:
    def __init__(self, k=2, iterations=100):
        self.k = k
        self.iters = iterations
        self.centroids = []
    def fit(self, data):
        self.centroids = random.sample(data, self.k)
        for _ in range(self.iters):
            clusters = [[] for __ in range(self.k)]
            for point in data:
                dists = [math.dist(point, c) for c in self.centroids]
                cluster_idx = dists.index(min(dists))
                clusters[cluster_idx].append(point)
            for idx in range(self.k):
                if clusters[idx]:
                    xs = [p[0] for p in clusters[idx]]
                    ys = [p[1] for p in clusters[idx]]
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    self.centroids[idx] = (cx, cy)
    def predict(self, point):
        dists = [math.dist(point, c) for c in self.centroids]
        return dists.index(min(dists))

def squared_error(y_true, y_pred):
    return sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred))

def accuracy_classification(y_true, y_pred):
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / len(y_true) if y_true else 0

def parse_html_title(html):
    pattern = r'<title>(.*?)</title>'
    matches = re.findall(pattern, html, re.IGNORECASE)
    if matches:
        return matches[0]
    return None

def parse_html_links(html):
    pattern = r'href=["\'](.*?)["\']'
    return re.findall(pattern, html)

def simple_web_scraper(url):
    try:
        data = small_web_request(url)
        title = parse_html_title(data)
        links = parse_html_links(data)
        return title, links
    except:
        return None, []

def get_http_status_code(host, path):
    conn = http.client.HTTPConnection(host)
    conn.request('GET', path)
    resp = conn.getresponse()
    code = resp.status
    conn.close()
    return code

def generate_random_matrix(rows, cols):
    mat = []
    for _ in range(rows):
        row = [random.randint(0, 10) for __ in range(cols)]
        mat.append(row)
    return mat

def spiral_matrix_traversal(matrix):
    result = []
    if not matrix:
        return result
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while True:
        if left > right:
            break
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1
        if top > bottom:
            break
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if left > right:
            break
        for i in range(right, left - 1, -1):
            result.append(matrix[bottom][i])
        bottom -= 1
        if top > bottom:
            break
        for i in range(bottom, top - 1, -1):
            result.append(matrix[i][left])
        left += 1
    return result

def combine_dictionaries(dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            result[k] = result.get(k, 0) + v
    return result

def partial_shuffle_string(s):
    arr = list(s)
    half = len(arr)//2
    random.shuffle(arr[:half])
    return "".join(arr)

def generate_random_date(start_year=2000, end_year=2025):
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    if month == 2:
        day = random.randint(1, 28)
    elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
    else:
        day = random.randint(1, 31)
    return datetime(year, month, day)

def parse_date_string(s, fmt="%Y-%m-%d"):
    try:
        return datetime.strptime(s, fmt)
    except:
        return None

def random_bytes_array(size):
    return os.urandom(size)

def random_float_list(n):
    return [random.random() for _ in range(n)]

def boolean_invert(value):
    return not value

def string_invert_case(s):
    return "".join(ch.lower() if ch.isupper() else ch.upper() for ch in s)

def simple_tokenizer(s):
    return re.findall(r'\w+', s)

def count_tokens_in_string(s):
    return len(simple_tokenizer(s))

def hex_to_int(hex_str):
    return int(hex_str, 16)

def int_to_hex(i):
    return hex(i)[2:]

def base_converter(num, base=2):
    if num == 0:
        return '0'
    digits = "0123456789ABCDEF"
    result = []
    n = abs(num)
    while n > 0:
        result.append(digits[n % base])
        n //= base
    if num < 0:
        result.append('-')
    return ''.join(reversed(result))

def find_duplicate_elements(lst):
    seen = set()
    duplicates = set()
    for x in lst:
        if x in seen:
            duplicates.add(x)
        else:
            seen.add(x)
    return list(duplicates)

class ArrayDisplayer:
    def display_vertical(self, arr):
        out = []
        for el in arr:
            out.append(str(el))
        return "\n".join(out)
    def display_horizontal(self, arr):
        return " ".join(str(el) for el in arr)

class SingleLinkedListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class SingleLinkedList:
    def __init__(self):
        self.head = None
    def insert_at_head(self, val):
        new_node = SingleLinkedListNode(val)
        new_node.next = self.head
        self.head = new_node
    def insert_at_tail(self, val):
        new_node = SingleLinkedListNode(val)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
    def find_value(self, val):
        current = self.head
        while current:
            if current.val == val:
                return True
            current = current.next
        return False
    def remove_value(self, val):
        if self.head is None:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next

def two_sum_problem(lst, target):
    d = {}
    for i, num in enumerate(lst):
        if target - num in d:
            return d[target - num], i
        d[num] = i
    return None

def linear_search(lst, val):
    for i, x in enumerate(lst):
        if x == val:
            return i
    return -1

def binary_search(lst, val):
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] == val:
            return mid
        elif lst[mid] < val:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def count_words_in_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = simple_tokenizer(text)
    return len(tokens)

class BasicCSVParser:
    def __init__(self, delimiter=','):
        self.delimiter = delimiter
    def parse(self, text):
        lines = text.split('\n')
        result = []
        for line in lines:
            if line.strip():
                result.append(line.split(self.delimiter))
        return result

def memory_efficient_count(path):
    c = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            c += 1
    return c

def filter_even_numbers(lst):
    return [x for x in lst if x % 2 == 0]

def filter_odd_numbers(lst):
    return [x for x in lst if x % 2 == 1]

def safe_dict_access(d, key, default=None):
    return d[key] if key in d else default

def repeated_in_string(s):
    counter = Counter(s)
    return [k for k, v in counter.items() if v > 1]

def double_factorial(n):
    if n <= 0:
        return 1
    return n * double_factorial(n-2)

def product_of_list(lst):
    p = 1
    for x in lst:
        p *= x
    return p

def random_sentence(word_count=5):
    words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    return " ".join(random.choice(words) for _ in range(word_count))

def string_contains_any(s, chars):
    return any(ch in s for ch in chars)

def average_list_values(lst):
    if not lst:
        return 0
    return sum(lst) / len(lst)

class DataCloner:
    def clone_list(self, lst):
        return lst[:]
    def clone_dict(self, d):
        return dict(d)
    def clone_set(self, s):
        return set(s)

def sliding_window_sum(lst, k):
    window_sum = sum(lst[:k])
    result = [window_sum]
    for i in range(k, len(lst)):
        window_sum += lst[i] - lst[i-k]
        result.append(window_sum)
    return result

def random_walk_1d(steps):
    position = 0
    path = [position]
    for _ in range(steps):
        step = random.choice([-1, 1])
        position += step
        path.append(position)
    return path

def parse_kv_string(s):
    lines = s.split('\n')
    d = {}
    for line in lines:
        if '=' in line:
            key, val = line.split('=', 1)
            d[key.strip()] = val.strip()
    return d

def generate_kv_string(d):
    lines = []
    for k, v in d.items():
        lines.append(f"{k}={v}")
    return "\n".join(lines)

def mix_two_lists(a, b):
    result = []
    for x, y in zip(a, b):
        result.append(x)
        result.append(y)
    return result

def alternate_upper_lower(s):
    result = []
    toggle = True
    for ch in s:
        if toggle:
            result.append(ch.upper())
        else:
            result.append(ch.lower())
        toggle = not toggle
    return "".join(result)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d, sep='.'):
    result = {}
    for k, v in d.items():
        parts = k.split(sep)
        current = result
        for p in parts[:-1]:
            if p not in current:
                current[p] = {}
            current = current[p]
        current[parts[-1]] = v
    return result

def random_uppercase_letters(n):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choice(chars) for _ in range(n))

def random_lowercase_letters(n):
    chars = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(chars) for _ in range(n))

def min_max_in_list(lst):
    if not lst:
        return None, None
    return min(lst), max(lst)

class ImportResolver3:
    def gather_imports(self, code_text):
        lines = code_text.split("\n")
        imported = []
        for line in lines:
            line = line.strip()
            if line.startswith("import "):
                part = line.split()[1]
                imported.append(part)
            elif line.startswith("from "):
                part = line.split()[1]
                imported.append(part)
        return list(set(imported))

class MultiLayerPerceptronMinimal:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = [[random.uniform(-1,1) for _ in range(hidden_size)] for __ in range(input_size)]
        self.b1 = [0]*hidden_size
        self.w2 = [[random.uniform(-1,1) for _ in range(output_size)] for __ in range(hidden_size)]
        self.b2 = [0]*output_size
    def activation(self, x):
        return 1/(1+math.exp(-x))
    def forward(self, inputs):
        hidden = []
        for j in range(len(self.w1[0])):
            s = 0
            for i in range(len(inputs)):
                s += inputs[i]*self.w1[i][j]
            s += self.b1[j]
            hidden.append(self.activation(s))
        out = []
        for k in range(len(self.w2[0])):
            s = 0
            for j in range(len(hidden)):
                s += hidden[j]*self.w2[j][k]
            s += self.b2[k]
            out.append(self.activation(s))
        return out

def read_binary_file(path):
    with open(path, 'rb') as f:
        return f.read()

def write_binary_file(path, data):
    with open(path, 'wb') as f:
        f.write(data)

def random_grid(rows, cols, val_range=(0, 9)):
    return [[random.randint(val_range[0], val_range[1]) for _ in range(cols)] for __ in range(rows)]

def sum_grid(grid):
    return sum(sum(row) for row in grid)

def parse_float_list_from_string(s):
    tokens = s.split()
    result = []
    for t in tokens:
        try:
            result.append(float(t))
        except:
            pass
    return result

def get_current_process_id():
    return os.getpid()

def sort_strings_by_length(strs):
    return sorted(strs, key=lambda x: len(x))

def wrap_text_at_length(s, length):
    words = s.split()
    lines = []
    current_line = ''
    for w in words:
        if len(current_line) + len(w) + 1 <= length:
            if current_line:
                current_line += ' '
            current_line += w
        else:
            lines.append(current_line)
            current_line = w
    if current_line:
        lines.append(current_line)
    return '\n'.join(lines)

def count_digit_occurrences(num):
    return Counter(str(num))

def fill_none_with_value(lst, val):
    return [val if x is None else x for x in lst]

def odd_even_partition(lst):
    odds = []
    evens = []
    for x in lst:
        if x % 2 == 0:
            evens.append(x)
        else:
            odds.append(x)
    return odds, evens

def nearest_square(num):
    root = int(math.sqrt(num))
    candidates = [root**2, (root+1)**2]
    return min(candidates, key=lambda x: abs(x - num))

def partial_censor_string(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, '*' * len(word), s)

class SigmoidNeuron:
    def __init__(self):
        self.weight = random.uniform(-1,1)
        self.bias = random.uniform(-1,1)
    def forward(self, x):
        return 1/(1+math.exp(-(self.weight*x + self.bias)))

def xor_cipher(s, key=42):
    return ''.join(chr(ord(ch) ^ key) for ch in s)

def approximate_euler(n_terms=10):
    return sum(1 / math.factorial(i) for i in range(n_terms))

def hex_md5_of_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()

def csv_sniffer_example(path):
    with open(path, 'r', encoding='utf-8') as f:
        sample = f.read(1024)
    dialect = csv.Sniffer().sniff(sample)
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, dialect)
        return list(reader)

def local_ip_address():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def random_select_words(words, k):
    return random.sample(words, k)

class BasicNetworkClient:
    def __init__(self):
        self.socket = None
    def connect(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
    def send(self, data):
        if self.socket:
            self.socket.sendall(data.encode('utf-8'))
    def receive(self):
        if self.socket:
            return self.socket.recv(4096).decode('utf-8')
    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None

def frequency_of_elements(lst):
    return Counter(lst)

def float_range(start, stop, step):
    while start < stop:
        yield start
        start += step

def random_rgb_color():
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

def multiple_import_resolver(code_block):
    found = []
    lines = code_block.splitlines()
    for line in lines:
        if 'import ' in line:
            parts = line.strip().split()
            if 'import' in parts:
                idx = parts.index('import')
                if idx + 1 < len(parts):
                    found.append(parts[idx+1])
        if 'from ' in line:
            parts = line.strip().split()
            if 'from' in parts:
                idx = parts.index('from')
                if idx + 1 < len(parts):
                    found.append(parts[idx+1])
    return list(set(found))

def multiply_string(s, times):
    return s * times

def read_url_headers(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as response:
        return response.info()

def CaesarCipher(s, shift=3):
    result = []
    for ch in s:
        if ch.isalpha():
            base = 'A' if ch.isupper() else 'a'
            result.append(chr((ord(ch) - ord(base) + shift) % 26 + ord(base)))
        else:
            result.append(ch)
    return ''.join(result)

def unCaesarCipher(s, shift=3):
    return CaesarCipher(s, -shift)

def matrix_transpose(matrix):
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    result = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(matrix[i][j])
        result.append(new_row)
    return result

def sum_of_diagonal(matrix):
    total = 0
    for i in range(min(len(matrix), len(matrix[0]))):
        total += matrix[i][i]
    return total

class NodeTree:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder_traversal(root):
    if not root:
        return []
    return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right)

def preorder_traversal(root):
    if not root:
        return []
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right)

def postorder_traversal(root):
    if not root:
        return []
    return postorder_traversal(root.left) + postorder_traversal(root.right) + [root.val]

def find_in_bst(root, val):
    current = root
    while current:
        if current.val == val:
            return True
        elif val < current.val:
            current = current.left
        else:
            current = current.right
    return False

def insert_into_bst(root, val):
    if not root:
        return NodeTree(val)
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    return root

def get_bst_min(root):
    if not root:
        return None
    while root.left:
        root = root.left
    return root.val

def get_bst_max(root):
    if not root:
        return None
    while root.right:
        root = root.right
    return root.val

def random_bst(num_nodes, value_range=(0,100)):
    values = [random.randint(value_range[0], value_range[1]) for _ in range(num_nodes)]
    root = None
    for v in values:
        root = insert_into_bst(root, v)
    return root

def reverse_file_lines(src, dst):
    lines = read_file_lines(src)
    lines.reverse()
    write_file_lines(dst, lines)

class BasicHashTable:
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]
    def hash_func(self, key):
        return hash(key) % self.size
    def set(self, key, value):
        idx = self.hash_func(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx][i] = (key, value)
                return
        self.table[idx].append((key, value))
    def get(self, key):
        idx = self.hash_func(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None
    def remove(self, key):
        idx = self.hash_func(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx].pop(i)
                return
    def keys(self):
        result = []
        for bucket in self.table:
            for k, _ in bucket:
                result.append(k)
        return result

class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, val):
        if self.root is None:
            self.root = NodeTree(val)
        else:
            insert_into_bst(self.root, val)
    def find(self, val):
        return find_in_bst(self.root, val)
    def get_min(self):
        return get_bst_min(self.root)
    def get_max(self):
        return get_bst_max(self.root)

def remove_duplicates_inplace(lst):
    i = 0
    while i < len(lst) - 1:
        if lst[i] == lst[i+1]:
            lst.pop(i+1)
        else:
            i += 1
    return lst

def compress_string(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(len(s)-1):
        if s[i] == s[i+1]:
            count += 1
        else:
            result.append(s[i])
            result.append(str(count))
            count = 1
    result.append(s[-1])
    result.append(str(count))
    return "".join(result)

def decompress_string(s):
    result = []
    i = 0
    while i < len(s):
        char = s[i]
        i += 1
        num_str = []
        while i < len(s) and s[i].isdigit():
            num_str.append(s[i])
            i += 1
        count = int("".join(num_str))
        result.append(char * count)
    return "".join(result)

def round_float_to_places(value, places):
    factor = 10**places
    return math.floor(value*factor + 0.5)/factor

def is_subsequence(s, t):
    it = iter(t)
    return all(c in it for c in s)

def full_justify(words, max_width):
    lines = []
    current_line = []
    current_length = 0
    for w in words:
        if current_length + len(w) + len(current_line) > max_width:
            for i in range(max_width - current_length):
                current_line[i % (len(current_line)-1 or 1)] += ' '
            lines.append(''.join(current_line))
            current_line, current_length = [], 0
        current_line.append(w)
        current_length += len(w)
    lines.append(' '.join(current_line).ljust(max_width))
    return lines

def generate_random_seeded_numbers(seed_val, n, start=0, end=100):
    r = random.Random(seed_val)
    return [r.randint(start, end) for _ in range(n)]

def day_of_week_from_date(dt):
    return dt.strftime('%A')

def partial_grid_extract(grid, row_start, row_end, col_start, col_end):
    return [r[col_start:col_end] for r in grid[row_start:row_end]]

def gcd_of_list(lst):
    g = lst[0]
    for x in lst[1:]:
        g = sequence_gcd(g, x)
    return g

def lcm_of_list(lst):
    def lcm(a, b):
        return abs(a*b) // sequence_gcd(a, b) if a and b else 0
    current = lst[0]
    for x in lst[1:]:
        current = lcm(current, x)
    return current

def list_to_str(lst, delimiter=" "):
    return delimiter.join(str(x) for x in lst)

def substring_occurrences(s, sub):
    c = start = 0
    while True:
        start = s.find(sub, start)
        if start == -1:
            return c
        c += 1
        start += len(sub)

class AStarSearch:
    def __init__(self):
        pass
    def heuristic(self, a, b):
        return math.dist(a, b)
    def search(self, start, goal, neighbors_func):
        open_set = [(0, start)]
        came_from = {start: None}
        g_score = {start: 0}
        while open_set:
            _, current = open_set.pop(0)
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            for neighbor in neighbors_func(current):
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    open_set.append((f_score, neighbor))
            open_set.sort(key=lambda x: x[0])
        return []

def approximate_pi(n_terms=1000):
    s = 0
    for i in range(n_terms):
        s += ((-1)**i)/(2*i+1)
    return 4*s

def reverse_each_word(s):
    words = s.split()
    return " ".join(word[::-1] for word in words)

def top_k_frequent(lst, k):
    freq = Counter(lst)
    return [x for x, _ in freq.most_common(k)]

def parse_config_file(path):
    result = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if '=' in line:
            key, val = line.split('=', 1)
            result[key.strip()] = val.strip()
    return result

def save_config_file(path, config_dict):
    lines = [f"{k}={v}" for k, v in config_dict.items()]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def power_mod(base, exponent, modulus):
    return pow(base, exponent, modulus)

def prime_sieve(n):
    sieve = [True]*(n+1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [x for x, val in enumerate(sieve) if val]

def approximate_ln_2(n_terms=100):
    return sum(1/i for i in range(1, n_terms+1)) - math.log(1)

def sum_of_series(lst):
    return sum(lst)

def product_of_series(lst):
    return product_of_list(lst)

def multi_import_finder(text):
    lines = text.split('\n')
    found = []
    for line in lines:
        if line.strip().startswith('import'):
            found.append(line.strip().split()[1])
        elif line.strip().startswith('from'):
            found.append(line.strip().split()[1])
    return found

class WeightedGraph:
    def __init__(self):
        self.adj = defaultdict(list)
    def add_edge(self, u, v, w):
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))
    def dijkstra(self, start):
        dist = {node: float('inf') for node in self.adj}
        dist[start] = 0
        visited = set()
        pq = [(0, start)]
        while pq:
            d, node = pq.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in self.adj[node]:
                nd = d + weight
                if nd < dist[neighbor]:
                    dist[neighbor] = nd
                    pq.append((nd, neighbor))
            pq.sort(key=lambda x: x[0])
        return dist

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def random_ascii_string(length):
    chars = [chr(i) for i in range(32, 127)]
    return ''.join(random.choice(chars) for _ in range(length))

def repeated_substring_pattern(s):
    ss = (s+s)[1:-1]
    return ss.find(s) != -1

def dynamic_fibonacci(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = dynamic_fibonacci(n-1, cache) + dynamic_fibonacci(n-2, cache)
    return cache[n]

def matrix_trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))

def diagonal_difference(matrix):
    d1 = sum(matrix[i][i] for i in range(len(matrix)))
    d2 = sum(matrix[i][len(matrix)-i-1] for i in range(len(matrix)))
    return abs(d1 - d2)

def chunk_string(s, size):
    return [s[i:i+size] for i in range(0, len(s), size)]

def is_string_rotation(s1, s2):
    if len(s1) != len(s2):
        return False
    return s2 in (s1+s1)

class BFSGraph:
    def __init__(self):
        self.adj = defaultdict(list)
    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        result = []
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in self.adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return result

def time_function_call(func, *args, **kwargs):
    start = datetime.now()
    result = func(*args, **kwargs)
    end = datetime.now()
    return result, (end - start).total_seconds()

class BasicLogisticRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = 0
    def sigmoid(self, z):
        return 1/(1+math.exp(-z))
    def fit(self, X, y):
        n = len(X)
        self.w = 0
        for _ in range(self.iters):
            dw = 0
            db = 0
            for xi, yi in zip(X, y):
                z = self.w * xi + self.b
                pred = self.sigmoid(z)
                dw += (pred - yi)*xi
                db += (pred - yi)
            self.w -= self.lr * dw / n
            self.b -= self.lr * db / n
    def predict_proba(self, X):
        return [self.sigmoid(self.w*xi + self.b) for xi in X]
    def predict(self, X):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

def build_adjacency_matrix(edges, n_nodes):
    matrix = [[0]*n_nodes for _ in range(n_nodes)]
    for (u, v) in edges:
        matrix[u][v] = 1
        matrix[v][u] = 1
    return matrix

def adjacency_matrix_to_list(matrix):
    adj_list = defaultdict(list)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val == 1:
                adj_list[i].append(j)
    return dict(adj_list)

class BasicNaiveBayes:
    def __init__(self):
        self.classes = []
        self.log_priors = {}
        self.word_counts = {}
        self.class_word_totals = {}
    def fit(self, X, y):
        self.classes = list(set(y))
        for c in self.classes:
            self.log_priors[c] = math.log(y.count(c)/len(y))
            self.word_counts[c] = defaultdict(int)
            self.class_word_totals[c] = 0
        for text, label in zip(X, y):
            for word in text.split():
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
    def predict(self, X):
        preds = []
        for text in X:
            class_scores = {}
            for c in self.classes:
                score = self.log_priors[c]
                for word in text.split():
                    count = self.word_counts[c].get(word, 0)
                    score += math.log((count+1)/(self.class_word_totals[c]+len(self.word_counts[c])))
                class_scores[c] = score
            preds.append(max(class_scores, key=class_scores.get))
        return preds

def array_intersection(a, b):
    set_a = set(a)
    set_b = set(b)
    return list(set_a & set_b)

def array_union(a, b):
    return list(set(a) | set(b))

def array_diff(a, b):
    return list(set(a) - set(b))

class DataCollector:
    def __init__(self):
        self.records = []
    def add_record(self, rec):
        self.records.append(rec)
    def get_records(self):
        return self.records
    def sort_records(self, key):
        self.records.sort(key=key)

def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return None
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def partial_subsequence(lst, length):
    return lst[:length]

def sliding_window_maximum(lst, k):
    dq = deque()
    result = []
    for i, val in enumerate(lst):
        while dq and dq[-1][1] <= val:
            dq.pop()
        dq.append((i, val))
        if dq[0][0] <= i-k:
            dq.popleft()
        if i >= k-1:
            result.append(dq[0][1])
    return result

def sqrt_newton(x, tolerance=1e-7):
    if x < 0:
        return None
    guess = x/2.0
    while True:
        new_guess = 0.5*(guess + x/guess)
        if abs(new_guess - guess) < tolerance:
            return new_guess
        guess = new_guess

def fisher_yates_shuffle(lst):
    for i in range(len(lst)-1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
    return lst

def parse_python_imports(code):
    lines = code.split("\n")
    modules = []
    for l in lines:
        if l.startswith("import ") or l.startswith("from "):
            modules.append(l)
    return modules

class WeightedQuickUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1]*n
    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
    def connected(self, p, q):
        return self.find(p) == self.find(q)

def triangle_type(a, b, c):
    if a+b <= c or b+c <= a or c+a <= b:
        return "Invalid"
    if a == b == c:
        return "Equilateral"
    if a == b or b == c or c == a:
        return "Isosceles"
    return "Scalene"

def read_last_n_lines(path, n):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines[-n:] if n <= len(lines) else lines

def safe_division(a, b):
    try:
        return a / b
    except:
        return None

def parse_integers_in_brackets(s):
    pattern = r'\[(\d+)\]'
    return [int(x) for x in re.findall(pattern, s)]

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
            result.append((prev, count))
            prev = s[i]
            count = 1
    result.append((prev, count))
    return result

def decode_run_length(pairs):
    return "".join(ch * cnt for ch, cnt in pairs)

class RedBlackDummy:
    def __init__(self):
        self.data = []
    def insert(self, value):
        self.data.append(value)
        self.data.sort()
    def search(self, value):
        return value in self.data

def parse_url_parameters(url):
    query = urllib.parse.urlparse(url).query
    return dict(urllib.parse.parse_qsl(query))

def read_file_in_chunks(path, chunk_size=1024):
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def generate_file_md5(path):
    hasher = hashlib.md5()
    for chunk in read_file_in_chunks(path, 4096):
        hasher.update(chunk)
    return hasher.hexdigest()

def palindrome_partition(s):
    result = []
    part = []
    def backtrack(start):
        if start == len(s):
            result.append(part[:])
            return
        for end in range(start, len(s)):
            substring = s[start:end+1]
            if substring == substring[::-1]:
                part.append(substring)
                backtrack(end+1)
                part.pop()
    backtrack(0)
    return result

class BasicRNN:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.w_xh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for __ in range(input_size)]
        self.w_hh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for __ in range(hidden_size)]
        self.b_h = [0]*hidden_size
    def step(self, x, h):
        new_h = []
        for j in range(self.hidden_size):
            sum_inp = 0
            for i in range(len(x)):
                sum_inp += x[i]*self.w_xh[i][j]
            for i in range(self.hidden_size):
                sum_inp += h[i]*self.w_hh[i][j]
            sum_inp += self.b_h[j]
            new_h.append(math.tanh(sum_inp))
        return new_h
    def forward_sequence(self, sequence):
        h = [0]*self.hidden_size
        for x in sequence:
            h = self.step(x, h)
        return h

def max_contiguous_sum(lst):
    max_so_far = lst[0]
    current = lst[0]
    for i in range(1, len(lst)):
        current = max(lst[i], current+lst[i])
        max_so_far = max(max_so_far, current)
    return max_so_far

def windchill_calculation(temp_celsius, wind_speed_kmh):
    temp_f = temp_celsius*9/5+32
    speed_mph = wind_speed_kmh * 0.621371
    wc = 35.74 + (0.6215*temp_f) - 35.75*(speed_mph**0.16) + 0.4275*temp_f*(speed_mph**0.16)
    return (wc - 32)*5/9

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def guess_file_encoding(path):
    with open(path, 'rb') as f:
        raw = f.read(4)
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        return 'utf-16'
    elif raw.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    else:
        return 'utf-8'

def triangular_number(n):
    return n*(n+1)//2

def hexagonal_number(n):
    return n*(2*n-1)

def pentagonal_number(n):
    return n*(3*n-1)//2

def nth_harmonic(n):
    return sum(1/i for i in range(1, n+1))

def tokenize_code_snippet(snippet):
    pattern = r"[A-Za-z_][A-Za-z0-9_]*"
    return re.findall(pattern, snippet)

class SimpleTokenizerVariant:
    def __init__(self):
        self.pattern = r"[A-Za-z_][A-Za-z0-9_]*"
    def tokenize(self, snippet):
        return re.findall(self.pattern, snippet)

def approximate_cosine(x, terms=10):
    s = 0
    for n in range(terms):
        s += ((-1)**n)*(x**(2*n))/math.factorial(2*n)
    return s

def approximate_sine(x, terms=10):
    s = 0
    for n in range(terms):
        s += ((-1)**n)*(x**(2*n+1))/math.factorial(2*n+1)
    return s

def gcd_extended(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = gcd_extended(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTreeOps:
    def height(self, root):
        if not root:
            return 0
        return 1 + max(self.height(root.left), self.height(root.right))
    def count_nodes(self, root):
        if not root:
            return 0
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)
    def level_order(self, root):
        results = []
        if not root:
            return results
        queue = deque([root])
        while queue:
            node = queue.popleft()
            results.append(node.data)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return results

def random_permutation(n):
    arr = list(range(n))
    random.shuffle(arr)
    return arr

def second_smallest(lst):
    if len(lst) < 2:
        return None
    smallest, second = (float('inf'), float('inf'))
    for x in lst:
        if x < smallest:
            second = smallest
            smallest = x
        elif x < second and x != smallest:
            second = x
    return second if second != float('inf') else None

def stable_marriage_problem(men_prefs, women_prefs):
    free_men = list(men_prefs.keys())
    next_proposal = {m: 0 for m in men_prefs}
    engaged = {}
    while free_men:
        man = free_men.pop(0)
        woman = men_prefs[man][next_proposal[man]]
        next_proposal[man] += 1
        if woman not in engaged:
            engaged[woman] = man
        else:
            current_man = engaged[woman]
            if women_prefs[woman].index(man) < women_prefs[woman].index(current_man):
                engaged[woman] = man
                free_men.append(current_man)
            else:
                free_men.append(man)
    return {v: k for k, v in engaged.items()}

def type_of_triangle_by_sides(a, b, c):
    if a + b <= c or b + c <= a or a + c <= b:
        return "Not a triangle"
    if a == b == c:
        return "Equilateral"
    if a == b or b == c or a == c:
        return "Isosceles"
    return "Scalene"

def reverse_bits(n):
    b = bin(n)[2:][::-1]
    return int(b, 2)

def is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.sqrt(n))
    return r*r == n

class BasicCNNPlaceholder:
    def __init__(self, input_dim, filter_count):
        self.input_dim = input_dim
        self.filter_count = filter_count
    def forward(self, inputs):
        outputs = []
        for _ in range(self.filter_count):
            outputs.append([sum(x) for x in inputs])
        return outputs

def sort_dict_by_value(d):
    return dict(sorted(d.items(), key=lambda x: x[1]))

def limit_file_size(path, max_size):
    size = os.path.getsize(path)
    if size <= max_size:
        return
    with open(path, 'rb') as f:
        data = f.read(max_size)
    with open(path, 'wb') as f:
        f.write(data)

def empty_directory_contents(path):
    for item in os.listdir(path):
        p = os.path.join(path, item)
        if os.path.isfile(p):
            os.remove(p)
        else:
            remove_directory(p)

def random_lower_upper_string(n):
    s = []
    for _ in range(n):
        c = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        s.append(c)
    return "".join(s)

def matrix_rotate_90(matrix):
    return list(zip(*matrix[::-1]))

def apply_mask_to_string(s, mask):
    r = []
    for ch, m in zip(s, mask):
        if m == '1':
            r.append(ch.upper())
        else:
            r.append(ch.lower())
    return "".join(r)

class MultiImportResolverVariant:
    def resolve_imports_variant(self, code_str):
        lines = code_str.split('\n')
        found = set()
        for l in lines:
            if l.strip().startswith('import') or l.strip().startswith('from'):
                parts = l.strip().split()
                if len(parts) > 1:
                    found.add(parts[1])
        return list(found)

def accumulate_list(lst):
    total = 0
    result = []
    for x in lst:
        total += x
        result.append(total)
    return result

def matrix_scalar_multiply(matrix, scalar):
    return [[scalar*val for val in row] for row in matrix]

def simple_smtp_mock_server(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)
    conn, addr = srv.accept()
    conn.send(b'220 mock smtp server ready\r\n')
    data = conn.recv(1024)
    conn.send(b'250 OK\r\n')
    conn.close()
    srv.close()

def replace_characters(s, replacements):
    arr = list(s)
    for idx, rep in replacements:
        if idx < len(arr):
            arr[idx] = rep
    return "".join(arr)

def numeric_range(start, end):
    return list(range(start, end))

def crypt_shift_string(s, shift):
    return "".join(chr((ord(ch) + shift) % 256) for ch in s)

def password_strength_estimator(password):
    score = 0
    if re.search(r'[A-Z]', password):
        score += 1
    if re.search(r'[a-z]', password):
        score += 1
    if re.search(r'[0-9]', password):
        score += 1
    if re.search(r'[^A-Za-z0-9]', password):
        score += 1
    if len(password) >= 8:
        score += 1
    return score

def rest_api_like_call(host, endpoint):
    conn = http.client.HTTPConnection(host)
    conn.request('GET', endpoint)
    resp = conn.getresponse()
    data = resp.read().decode('utf-8')
    conn.close()
    return data
