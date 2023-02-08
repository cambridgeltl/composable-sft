from collections import defaultdict
import os
import sys

from datasets import load_dataset
import editdistance
from nltk.tokenize.treebank import TreebankWordDetokenizer

OUTPUT_DIR = 'data/smsa-trimmed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

smsa = load_dataset('indonlp/indonlu', 'smsa')
nusa = load_dataset('indonlp/NusaX-senti', 'ind')

detokenizer = TreebankWordDetokenizer()

smsa_text = [
    (detokenizer.detokenize(s.split()), (i, name))
    for name, split in smsa.items()
    for i, s in enumerate(split['text'])
]
nusa_text = [
    s.lower()
    for s in nusa['test']['text']
]

unfound = 0
match_ids = set()
for nt in nusa_text:
    smallest_distance, best_candidate, best_candidate_id = min(
        (editdistance.eval(nt, candidate), candidate, candidate_id)
        for candidate, candidate_id in smsa_text
    )
    match_ids.add(best_candidate_id)
    sys.stderr.write(f'({smallest_distance}) Mapped "{nt}" -> "{best_candidate}"\n')


# SMSA and NusaX-senti use different label conventions; maps SMSA to NusaX-senti convention
LABEL_MAP = {
    0: 2,
    1: 1,
    2: 0,
}
for split_name, split in smsa.items():
    with open(os.path.join(OUTPUT_DIR, f'{split_name}.tsv'), 'w') as f:
        f.write('text\tlabel\n')
        for i, example in enumerate(split):
            if (i, split_name) in match_ids:
                sys.stderr.write(f'Omitting {split_name} {i}\n')
                continue
            text = example['text']
            label = LABEL_MAP[example['label']]
            f.write(f'{text}\t{label}\n')
