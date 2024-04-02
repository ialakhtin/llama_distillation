import os
import argparse
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file-name', default='wiki_data.json')
    parser.add_argument('--add-statistics', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    texts = pq.read_table(args.input_file)['text'].to_numpy()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_file = open(os.path.join(dir_path, f'../processed_data/{args.output_file_name}'), 'w')

    total_len = 0
    token_count = 0
    for text in tqdm(texts):
        if args.add_statistics:
            total_len += len(text)
            token_count += len(word_tokenize(text))
        out_file.write(json.dumps({'content': text})+'\n')
    if args.add_statistics:
        print(f"Total len: {total_len}, token count: {token_count}")

if __name__ == '__main__':
    main()