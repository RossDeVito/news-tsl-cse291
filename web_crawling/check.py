import json
import os


def parse_keyword(file):
    with open(file, 'r') as f:
        kw = json.load(f)
    return kw

if __name__ == '__main__':
    api_key = '4ff3ddcd-148a-4cbe-83f9-682aa211c090'
    root = './datasets/'

    tl_dirs = [os.path.join(root, name) for name in os.listdir(root) if
                  os.path.isdir(os.path.join(root, name))]
    for tl in tl_dirs:
        topic = [os.path.join(tl, name) for name in os.listdir(tl) if
                   os.path.isdir(os.path.join(tl, name))]
        for t in topic:
            index = 0
            with open(t+'/articles.jsonl', 'r') as f:
                for l in f:
                    index += 1
            print(t)
            print(index)