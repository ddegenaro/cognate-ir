import math
import argparse
from typing import Callable, Union

import pandas as pd
from pybktree import BKTree
from strsimpy import (
    Levenshtein,
    WeightedLevenshtein,
    Damerau,
    OptimalStringAlignment,
    LongestCommonSubsequence,
    NGram,
    QGram,
)

def fast_int_round(fn: Callable) -> Callable:
    
    """
    Accepts a string distance function and returns an integer-valued version that is maximally
    faithful to the original function.
    
    Args:
        - `fn: Callable` - the function.
        
    Returns:
        - `Callable` - the integer-valued version of `fn`.
    """
    
    return lambda a, b: round(fn(a, b))

def get_tree(name: str) -> BKTree:
    
    """
    Returns a `BKTree` using the specified string distance function.
    
    Args:
        - `name: str` - The alias of the string distance function to use.
        
    Returns:
        - `BKTree` - The tree with items from `docs.tsv`.
    """
    
    global docs
    
    tree = BKTree(
        distance_func=rounded_dists[name]
    )
    l = len(docs['doc'])
    for i, item in enumerate(docs['doc']):
        tree.add(item)
        if i % 1_000 == 0:
            print(f'Loading tree \'{name}\'...{round(100 * i / l)}%', end='\r')
    return tree

def search(
    name: str,
    split: Union[str, pd.DataFrame],
    min_R: int,
    max_R: int,
    max_K: int,
    tree: BKTree = None
):
    
    """
    Searches a `BKTree` for all the queries in `split`. If `split` is `'train'` or `'test'`, it searches
    for the queries in the appropriate split. If `split` is a `pd.DataFrame` containing queries, it
    processes those. The `BKTree` is created if it does not exist. Writes a `.jsonl` result file to
    `searches/`.
    
    Args:
        - `name: str` - The string distance function to use for the BKTree. Ignored if `tree` is passed,
        but used for file-naming regardless.
        - `split: Union[str, pd.DataFrame]` - The data split whose queries are to be used.
        - `min_R: int` - A starting guess radius to be used.
        - `max_R: int` - The largest radius to use.
        - `max_K: int` - The largest number of matches to return for each query.
        - `tree: BKTree` - The tree to search over. If `None`, it is created based on `name`.
    """
    
    global train_queries
    global test_queries
    global docs
    global doc_index
    
    if split == 'train':
        queries = train_queries
    elif split == 'test':
        queries = test_queries
    elif type(split) == pd.DataFrame:
        queries = split
        split = f'custom{len(split)}'
        
    if tree is None:
        tree = get_tree(name)
        
    num_queries = len(queries)
    log_freq = math.ceil(num_queries / 100)

    run = []

    for i, row in queries.iterrows():
        
        query_id = row['query_id']
        query = row['query']
        results = []
        
        previous_len = 0
        for R in range(min_R, max_R + 1): # keep searching larger radii
            more = tree.find(query, R)
            more = more[previous_len:]
            # print(len(more))
            previous_len = len(more)
            for j in range(1, len(more)):
                score, doc = more[j]
                doc_ids = doc_index.get(doc, [])
                for doc_id in doc_ids:
                    results.append({
                        'query_id': query_id,
                        'doc_id': doc_id,
                        'score': 1. / score # inverse distance
                    })
                    if len(results) >= max_K:
                        break
                if len(results) >= max_K:
                    break
            if len(results) >= max_K:
                break
        results = results[:max_K]
        
        run.extend(results)
            
        if (i+1) % log_freq == 0:
            print(f'Processed query {i+1}/{num_queries}...', end='\r')
        
    pd.DataFrame(run).to_json(
        f'searches/{name}_{split}_R_{min_R}_{max_R}_K_{max_K}.jsonl',
        lines=True,
        orient='records'
    )
    
if __name__ == '__main__':
    
    dists = {
        'lev': Levenshtein().distance,
        'wlev_ins': WeightedLevenshtein(
            insertion_cost_fn=lambda char: 2
        ).distance,
        'wlev_del': WeightedLevenshtein(
            deletion_cost_fn=lambda char: 2
        ).distance,
        'wlev_sub': WeightedLevenshtein(
            substitution_cost_fn=lambda char_a, char_b: 2
        ).distance,
        'dam': Damerau().distance,
        'osa': OptimalStringAlignment().distance,
        'lcs': LongestCommonSubsequence().distance,
        'ng1': NGram(1).distance,
        'ng2': NGram(2).distance,
        'ng3': NGram(3).distance,
        'ng4': NGram(4).distance,
        'qg1': QGram(1).distance,
        'qg2': QGram(2).distance,
        'qg3': QGram(3).distance,
        'qg4': QGram(4).distance
    }
    
    rounded_dists = {
        name: fast_int_round(dist_fn) for name, dist_fn in dists.items()
    }

    for name in dists:
        assert round(dists[name]('hello', 'hallo')) == rounded_dists[name]('hello', 'hallo')
        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        '-n',
        choices = list(dists.keys()),
        required=True
    )
    parser.add_argument(
        '--split',
        '-s',
        choices = ['train', 'test'],
        required=True
    )
    parser.add_argument(
        '--min_R',
        '-m',
        type = int,
        default = 3
    )
    parser.add_argument(
        '--max_R',
        '-M',
        type = int,
        default = 10
    )
    parser.add_argument(
        '--max_K',
        '-k',
        '-K',
        type = int,
        default = 100
    )
        
    train_queries = pd.read_csv('queries-train.tsv', sep='\t')
    test_queries = pd.read_csv('queries-test.tsv', sep='\t')
    docs = pd.read_csv('docs.tsv', sep='\t')
    doc_index = docs.groupby('doc')['doc_id'].unique().to_dict()
    
    args = parser.parse_args()
    
    search(
        name = args.name,
        split = args.split,
        min_R = args.min_R,
        max_R = args.max_R,
        max_K = args.max_K
    )