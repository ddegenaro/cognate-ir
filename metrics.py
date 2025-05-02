import os
import json

import pandas as pd
import ir_measures as irms
from ir_measures import MAP, R, P

REWRITE = True

measures = [
    R@1, R@5, R@10, R@20, R@50, R@100, R@500, R@1000,
    P@1, P@5, P@10, P@20, P@50, P@100, P@500, P@1000,
    MAP
]

qrels_train = None
langs_train = None
qrels_test = None
langs_test = None

os.makedirs('metrics', exist_ok=True)
os.makedirs('metrics/all', exist_ok=True)

for search in os.listdir('searches'):
    
    if 'train' in search and qrels_train is None:
        qrels_train = pd.DataFrame(
            [json.loads(l) for l in open('qrels-train-v2.jsonl', 'r', encoding='utf-8')]
        )
        langs_train = qrels_train['query_lang'].unique()
        qrels = qrels_train
        langs = langs_train
    elif 'train' in search:
        qrels = qrels_train
        langs = langs_train
    elif 'test' in search and qrels_test is None:
        qrels_test = pd.DataFrame(
            [json.loads(l) for l in open('qrels-test-v2.jsonl', 'r', encoding='utf-8')]
        )
        langs_test = qrels_test['query_lang'].unique()
        qrels = qrels_test
        langs = langs_test
    elif 'test' in search:
        qrels = qrels_test
        langs = langs_test
    else:
        raise ValueError('Search is not over train or test queries.')
    
    results = pd.DataFrame(
        [json.loads(l) for l in open(f'searches/{search}', 'r', encoding='utf-8')]
    )
    
    results['query_id'] = results['query_id'].astype(str)
    results['doc_id'] = results['doc_id'].astype(str)
    
    all_scores = irms.calc_aggregate(
        measures,
        qrels,
        results
    )
    measures_dict = {}
    for key in sorted(all_scores, key=lambda x: x.__str__()):
        measures_dict[key.__str__()] = all_scores[key]
    ALL_PATH = os.path.join('metrics', 'all', f'{search.replace('jsonl', 'json')}')
    if not os.path.exists(ALL_PATH) or REWRITE:
        with open(
            ALL_PATH, 'w+', encoding='utf-8'
        ) as write_file:
            json.dump(measures_dict, write_file, indent=4)
            print(ALL_PATH)
    
    for lang in langs:
        PATH_TO_WRITE = os.path.join('metrics', lang, f'{search.replace('jsonl', 'json')}')
        if os.path.exists(PATH_TO_WRITE) and not REWRITE:
            continue
        qrels_lang = qrels[qrels['query_lang'] == lang]
        measures_dict = {}
        lang_scores = irms.calc_aggregate(
            measures,
            qrels_lang,
            results
        )
        for key in sorted(lang_scores, key=lambda x: x.__str__()):
            measures_dict[key.__str__()] = lang_scores[key]
        os.makedirs(f'metrics/{lang}', exist_ok=True)
        with open(
            PATH_TO_WRITE, 'w+', encoding='utf-8'
        ) as write_file:
            json.dump(measures_dict, write_file, indent=4)
            print(PATH_TO_WRITE)
