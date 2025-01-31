# src/test_queries.py

test_queries = [
    {
        "query": "Which function can find all the import statements in Python code?",
        "description": "Parsing imports with resolvers",
        "expected_keywords": ["import", "resolver", "find_imports", "ImportResolver"]
    },
    {
        "query": "How do I reverse the words in a sentence for string manipulation?",
        "description": "Reverse words in a string",
        "expected_keywords": ["reverse_words_in_string", "split", "string manipulation"]
    },
    {
        "query": "I need a basic function for linear regression that can train and predict values.",
        "description": "Simple linear regression",
        "expected_keywords": ["BasicRegressionModel", "fit", "predict"]
    },
    {
        "query": "Which snippet shows a method to insert and select rows from an SQLite database?",
        "description": "SQLite insertion and retrieval",
        "expected_keywords": ["INSERT INTO", "SELECT", "sqlite3", "simple_database_insert", "simple_database_fetch_all"]
    },
    {
        "query": "How do I build a random BST with a specified number of nodes?",
        "description": "Random BST creation",
        "expected_keywords": ["random_bst", "NodeTree", "insert_into_bst"]
    },
    {
        "query": "Where is the function that scrapes a webpage and returns the page title and links?",
        "description": "Web scraping with HTML parsing",
        "expected_keywords": ["simple_web_scraper", "parse_html_title", "parse_html_links"]
    },
    {
        "query": "I want to generate a random alphanumeric string of a given length.",
        "description": "Random string generation",
        "expected_keywords": ["random_alphanumeric_string", "random_hex_string"]
    },
    {
        "query": "Which function implements an XOR cipher for strings with a numeric key?",
        "description": "Basic XOR encryption",
        "expected_keywords": ["xor_cipher", "encryption", "string XOR"]
    },
    {
        "query": "I need a class for k-means clustering on 2D points. Where can I find it?",
        "description": "K-Means clustering",
        "expected_keywords": ["BasicKMeans", "fit", "predict", "centroids"]
    },
    {
        "query": "How do I compute a SHA256 hash of a given string?",
        "description": "SHA-256 string hashing",
        "expected_keywords": ["hash_string_sha256", "hashlib", "SHA256"]
    },
    {
        "query": "Which snippet can parse JSON strings and turn them into Python objects?",
        "description": "JSON string parsing",
        "expected_keywords": ["parse_json_string", "json.loads"]
    },
    {
        "query": "I want a function that sets up a basic TCP server to echo data in uppercase.",
        "description": "Simple socket server",
        "expected_keywords": ["simple_socket_server", "socket", "listen", "accept"]
    },
    {
        "query": "Show me the BFS graph code that returns nodes in breadth-first order.",
        "description": "BFS graph traversal",
        "expected_keywords": ["BFSGraph", "bfs", "deque", "adj"]
    },
    {
        "query": "How do I read a CSV file into a list or dictionary in Python?",
        "description": "Reading CSV files",
        "expected_keywords": ["read_csv_as_list", "read_csv_as_dicts", "csv"]
    },
    {
        "query": "Which function is responsible for compressing a string into run-length encoding?",
        "description": "String compression with run-length encoding",
        "expected_keywords": ["encode_run_length", "compress_string", "pairs"]
    },
    {
        "query": "Which snippet demonstrates searching for an item in a binary search tree?",
        "description": "BST value search",
        "expected_keywords": ["find_in_bst", "NodeTree", "BinarySearchTree"]
    },
    {
        "query": "Where is the code that censors a word in a string with asterisks?",
        "description": "Partial censor for words",
        "expected_keywords": ["partial_censor_string", "regex", "asterisks"]
    },
    {
        "query": "How do I shuffle a list in place using Fisher-Yates?",
        "description": "In-place list shuffling",
        "expected_keywords": ["fisher_yates_shuffle", "random.randint", "list shuffle"]
    },
    {
        "query": "I want a class that can log-prior and word-likelihoods for naive Bayes classification.",
        "description": "Naive Bayes classifier",
        "expected_keywords": ["BasicNaiveBayes", "log_priors", "word_counts", "predict"]
    },
    {
        "query": "Which snippet shows a function for splitting and flattening nested lists?",
        "description": "Flatten nested lists",
        "expected_keywords": ["flatten_nested_list", "isinstance", "recursive list"]
    },
    {
        "query": "I want to compute the GCD or LCM of an entire list of numbers. Which function does that?",
        "description": "GCD and LCM of lists",
        "expected_keywords": ["gcd_of_list", "lcm_of_list", "sequence_gcd"]
    },
    {
        "query": "How do I rotate a string to the left or right by a given number of characters?",
        "description": "String rotation",
        "expected_keywords": ["rotate_string_left", "rotate_string_right", "slicing"]
    },
    {
        "query": "Which snippet shows how to create a temporary file with a random name?",
        "description": "Creating temp files",
        "expected_keywords": ["create_temp_file", "os", "random.randint"]
    },
    {
        "query": "Where is the code for merging two dictionaries by summing their values?",
        "description": "Combine dictionaries",
        "expected_keywords": ["combine_dictionaries", "dict", "summing values"]
    },
    {
        "query": "I want to see the code for a minimal feed-forward MLP with random weights. Where is it?",
        "description": "Simple MultiLayer Perceptron",
        "expected_keywords": ["MultiLayerPerceptronMinimal", "activation", "forward", "hidden size"]
    }
]
