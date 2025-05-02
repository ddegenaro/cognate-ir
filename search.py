import math
import os
import json
import argparse
from time import time
from datetime import timedelta
from typing import Callable, Union

import pandas as pd
from pybktree import BKTree
from strsimpy import (
    Levenshtein,
    NormalizedLevenshtein,
    WeightedLevenshtein,
    Damerau,
    OptimalStringAlignment,
    LongestCommonSubsequence,
    MetricLCS,
    NGram,
    QGram,
    Cosine,
    Jaccard,
    SorensenDice
)

DISTS = {
    'lev': Levenshtein().distance,
    'nlev': NormalizedLevenshtein().distance,
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
    'mlcs': MetricLCS().distance,
    'ng1': NGram(1).distance,
    'ng2': NGram(2).distance,
    'ng3': NGram(3).distance,
    'ng4': NGram(4).distance,
    'qg1': QGram(1).distance,
    'qg2': QGram(2).distance,
    'qg3': QGram(3).distance,
    'qg4': QGram(4).distance,
    'cos1': Cosine(1).distance,
    'cos2': Cosine(2).distance,
    'cos3': Cosine(3).distance,
    'cos4': Cosine(4).distance,
    'jac1': Jaccard(1).distance,
    'jac2': Jaccard(2).distance,
    'jac3': Jaccard(3).distance,
    'jac4': Jaccard(4).distance,
    'dice1': SorensenDice(1).distance,
    'dice2': SorensenDice(2).distance,
    'dice3': SorensenDice(3).distance,
    'dice4': SorensenDice(4).distance
}

docs = pd.read_csv('docs.tsv', sep='\t')
doc_index = docs.groupby('doc')['doc_id'].unique().to_dict()

def get_queries(split: str) -> pd.DataFrame:
    return pd.read_csv(f'queries-{split}.tsv', sep='\t')

def fast_int_round(name: str, fn: Callable) -> Callable:
    
    """
    Accepts a string distance function and returns an integer-valued version that is maximally
    faithful to the original function.
    
    Args:
        - `name: str` - the name of the distance function.
        - `fn: Callable` - the distance function.
        
    Returns:
        - `Callable` - the integer-valued version of `fn`.
    """
    
    if (name in {'lev', 'dam', 'osa', 'lcs'}) or ('wlev' in name) or ('qg' in name):
        return lambda a, b: round(fn(a, b))
    else:
        return lambda a, b: round(100 * fn(a, b))

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
        distance_func=fast_int_round(name, DISTS[name])
    )
    l = len(docs['doc'])
    for i, item in enumerate(docs['doc']):
        tree.add(item)
        if i % 1_000 == 0:
            print(f'Loading tree \'{name}\'...{round(100 * i / l)}%', end='\r', flush=True)
    return tree

def search(
    name: str,
    split: Union[str, pd.DataFrame],
    min_R: int = 1,
    max_R: int = 100,
    max_K: int = 1000,
    tree: BKTree = None
) -> tuple[BKTree, pd.DataFrame]:
    
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
        
    Returns:
        - `BKTree` - THe tree that was just used in case it can be re-used.
    """
    
    fname = f'{name}_{split}_R_{min_R}_{max_R}_K_{max_K}.json'
    
    if os.path.exists(f'searches/{fname}l') and os.path.exists(f'timing/{fname}'):
        return None, None
    
    os.makedirs('searches', exist_ok=True)
    os.makedirs('timing', exist_ok=True)
    
    timing_info = {}
    
    global docs
    global doc_index
    
    if type(split) == str:
        if split == 'train' or split == 'test':
            queries = get_queries(split)
    elif type(split) == pd.DataFrame:
        queries = split
        split = f'custom{len(split)}'
    else:
        raise ValueError(f'{type(split)} is not a valid type to use as split. Must be str or pd.DataFrame.')
        
    if tree is None:
        tree_start = time()
        tree = get_tree(name)
        timing_info['tree_build_time'] = time() - tree_start
        timing_info['tree_build_time_per_item'] = timing_info['tree_build_time'] / len(docs['doc'])
        
    num_queries = len(queries)
    log_freq = math.ceil(num_queries / 100)

    run = []
    start = time()

    for i, row in queries.iterrows():
        
        query_id = row['query_id']
        query = row['query']
        results = []
        
        previous_len = 0
        for R in range(min_R, max_R + 1): # keep searching larger radii
            more = tree.find(query, R, max_K)
            more = more[previous_len:]
            # print(len(more))
            previous_len = len(more)
            for j in range(1, len(more)):
                score, doc = more[j]
                doc_ids = doc_index.get(doc, [])
                for doc_id in doc_ids:
                    try:
                        inv_score = 1. / score
                    except ZeroDivisionError:
                        inv_score = 1. / (score + 1e-4)
                    results.append({
                        'query_id': query_id,
                        'doc_id': doc_id,
                        'score': inv_score # inverse distance
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
            avg_time = (time() - start) / (i + 1)
            queries_remaining = num_queries - (i + 1)
            etr = str(timedelta(seconds = avg_time * queries_remaining))
            print(f'Processed query {i+1}/{num_queries}. ETR: {etr}.', end='\r', flush=True)
            
    timing_info['query_time'] = time() - start
    timing_info['query_time_per_query'] = timing_info['query_time'] / num_queries
    
    
    
    df = pd.DataFrame(run)
    df.to_json(
        f'searches/{fname}l',
        lines=True,
        orient='records'
    )
    
    json.dump(
        timing_info,
        open(f'timing/{fname}', 'w+', encoding='utf-8'),
        indent = 4
    )
    
    return tree, df
    
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        '-n',
        choices = list(DISTS.keys()),
        required=True
    )
    parser.add_argument(
        '--split',
        '-s',
        choices = ['train', 'test', 'both'],
        required=True
    )
    parser.add_argument(
        '--min_R',
        '-m',
        type = int,
        default = 1
    )
    parser.add_argument(
        '--max_R',
        '-M',
        type = int,
        default = 100 # might be safe to do this with new optimization
    )
    parser.add_argument(
        '--max_K',
        '-k',
        '-K',
        type = int,
        default = 1000
    )
    
    args = parser.parse_args()
    
    if args.split == 'both':
        tree, _ = search(
            name = args.name,
            split = 'train',
            min_R = args.min_R,
            max_R = args.max_R,
            max_K = args.max_K,
            tree = None
        )
        search(
            name = args.name,
            split = 'test',
            min_R = args.min_R,
            max_R = args.max_R,
            max_K = args.max_K,
            tree = tree # re-use
        )
    else:
        search(
            name = args.name,
            split = args.split,
            min_R = args.min_R,
            max_R = args.max_R,
            max_K = args.max_K,
            tree = None
        )