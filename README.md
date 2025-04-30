# Cognate IR

## Setup

1. Clone the repository and change to the project directory:

    ```bash
    git clone https://github.com/ddegenaro/cognate-ir.git
    cd cognate-ir
    ```

2. Clone the CogNet dataset repository, enter the CogNet directory, and unzip the dataset:

    ```bash
    git clone https://github.com/kbatsuren/CogNet.git
    cd CogNet
    unzip CogNet-v2.0.zip
    ```

3. Run all the cells in `make_qrels.ipynb` to reformat the data for IR. You may need to clean the original .tsv files by hand a bit, but it's not terribly difficult.

4. Create a Python environment however you like, activate it, and then install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

5. Search using any of the defined distance metrics via `search.py`.
   For example, to search over the training set using the Levenshtein distance, run:

    ```bash
    python search.py --name lev --split train
    ```

    By default, the BKTree will return the top 100 matches within a radius of 10.

    If you find it particularly slow, you can try increasing `--min_R`, which is the first guess for the radius needed to retrieve the top 100 matches. It will still return matches with radius less than `--min_R`.

    Note that there is no `--max_R` that guarantees 100 matches.

    If you need more or fewer matches, you can change `--max_K`.

    Distances:
    - `lev` = Levenshtein
    - `wlev_ins` = Weighted Levenshtein with higher insertion cost
    - `wlev_del` = Weighted Levenshtein with higher deletion cost
    - `wlev_sub` = Weighted Levenshtein with higher substitution cost
    - `dam` = Damerau-Levenshtein
    - `osa` = Optimal String Alignment
    - `lcs` = Longest Common Subsequence
    - `qg1`-`qg4` = QGrams with k=1-4
    - `ng1`-`ng4` = 1-grams thru 4-grams

    Argument aliases:
    - `--name` = Name of the distance metric to use (also `-n`)
    - `--split` = Which split to search over (train, dev, or test) (also `-s`)
    - `--min_R` = Minimum radius to search for matches (also `-m`)
    - `--max_R` = Maximum radius to search for matches (also `-M`)
    - `--max_K` = Maximum number of matches to return (also `-k` or `-K`)
