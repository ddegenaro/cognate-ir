import os
import csv
import json
import argparse

import pandas as pd
import ir_measures as irms
from ir_measures import MAP, R, P

def main(job_id: int, n_jobs: int):
    
    REWRITE = False

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
    
    for idx, search in enumerate(sorted(os.listdir('searches'))):
        
        if (idx + job_id) % n_jobs != 0:
            continue
        
        print(f'Processing search {search} at index {idx}.', flush=True)
        
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
        
        PATH_TO_AGG_ALL = os.path.join('metrics', 'all', f'{search.replace('jsonl', 'json')}')
        PATH_TO_PER_QUERY_ALL = os.path.join('metrics', 'all', f'{search.replace('jsonl', 'csv')}')
        
        if not (os.path.exists(PATH_TO_AGG_ALL) and os.path.exists(PATH_TO_PER_QUERY_ALL)) or REWRITE:
            all_scores = irms.calc_aggregate(
                measures,
                qrels,
                results
            )
            measures_dict = {}
            for key in sorted(all_scores, key=lambda x: x.__str__()):
                measures_dict[key.__str__()] = all_scores[key]

            with open(
                PATH_TO_AGG_ALL, 'w+', encoding='utf-8'
            ) as write_file:
                json.dump(measures_dict, write_file, indent=4)
                print(PATH_TO_AGG_ALL, flush=True)
                
            all_iter_calc = irms.iter_calc(
                measures,
                qrels,
                results
            )
            writer = csv.writer(open(PATH_TO_PER_QUERY_ALL, 'w+', encoding='utf-8'), delimiter='\t')
            writer.writerow(['query_id', 'measure', 'value'])
            for calc in all_iter_calc:
                writer.writerow([calc.query_id, calc.measure.__str__(), str(calc.value)])
            print(PATH_TO_PER_QUERY_ALL, flush=True)
        
        for lang in langs:
            PATH_TO_AGG_LANG = os.path.join('metrics', lang, f'{search.replace('jsonl', 'json')}')
            PATH_TO_PER_QUERY_LANG = os.path.join('metrics', lang, f'{search.replace('jsonl', 'csv')}')
            
            if (os.path.exists(PATH_TO_AGG_LANG) and os.path.exists(PATH_TO_PER_QUERY_LANG)) and not REWRITE:
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
                PATH_TO_AGG_LANG, 'w+', encoding='utf-8'
            ) as write_file:
                json.dump(measures_dict, write_file, indent=4)
                print(PATH_TO_AGG_LANG, flush=True)
            
            lang_iter_calc = irms.iter_calc(
                measures,
                qrels_lang,
                results
            )
            writer = csv.writer(open(PATH_TO_PER_QUERY_LANG, 'w+', encoding='utf-8'), delimiter='\t')
            writer.writerow(['query_id', 'measure', 'value'])
            for calc in lang_iter_calc:
                writer.writerow([calc.query_id, calc.measure.__str__(), str(calc.value)])
            print(PATH_TO_PER_QUERY_LANG, flush=True)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', '-j', type=int, required=True)
    parser.add_argument('--n_jobs', '-n', type=int, required=False, default=3)
    args = parser.parse_args()
    main(args.job_id, args.n_jobs)