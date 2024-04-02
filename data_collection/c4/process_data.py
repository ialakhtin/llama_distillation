import os
import random
import argparse
import json
import gzip
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-file-name', default='c4_data.json')
    parser.add_argument('--add-statistics', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_file = open(os.path.join(dir_path, f'../processed_data/{args.output_file_name}'), 'w')

    total_len = 0
    token_count = 0
    for file in input_files:
        if file.rsplit('.', 1)[-1] != 'gz':
            continue
        with gzip.open(file, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            if line.strip() == '':
                continue
            line_json = json.loads(line.strip())
            content = line_json['text']
            if args.add_statistics:
                total_len += len(content)
                token_count += len(word_tokenize(content))
            out_file.write(json.dumps({'content': content}) + '\n')
    if args.add_statistics:
        print(f"Total len: {total_len}, token count: {token_count}")

if __name__ == '__main__':
    main()