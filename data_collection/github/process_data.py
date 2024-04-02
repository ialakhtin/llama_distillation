import os
import argparse
import re
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize


C_LIKE_COMMENTS = ['java', 'js', 'cs', 'h', 'php', 'cpp', 'go', 'c', 'cc', 'hpp', 'scala']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file-name', default='github_data.json')
    parser.add_argument('--add-statistics', action='store_true')
    return parser.parse_args()

def get_output_file_path(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.normpath(os.path.join(dir_path, f'../processed_data/{name}'))

def remove_c_like_comments(content):
    content = re.sub(r'/\*.*\*/', '\n', content)
    return re.sub(r'//.*\n', '\n', content)

def remove_python_comments(content):
    return re.sub(r'#.*\n', '\n', content) 

def remove_comments(content, ext):
    if ext == 'py':
        return remove_python_comments(content)
    if ext in C_LIKE_COMMENTS:
        return remove_c_like_comments(content)
    return content

def main():
    args = parse_args()
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    output_path = get_output_file_path(args.output_file_name)
    print(output_path)
    out_file = open(output_path, 'w')
    total_len = 0
    token_count = 0
    for line in tqdm(lines):
        if line.strip() == '':
            continue
        line_json = json.loads(line.strip())
        if 'content' not in line_json or 'ext' not in line_json:
            continue
        content = remove_comments(line_json['content'], line_json['ext'])
        content = re.sub('^\n+', '', content)
        content = re.sub('\n\n+', '\n\n', content)
        if args.add_statistics:
            total_len += len(content)
            token_count += len(word_tokenize(content))
        out_file.write(json.dumps({'content': content}) + '\n')
    out_file.close()
    if args.add_statistics:
        print(f"Total len: {total_len}, token count: {token_count}")

if __name__ == '__main__':
    main()