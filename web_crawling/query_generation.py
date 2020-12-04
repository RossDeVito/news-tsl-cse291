import os
import json
from datetime import datetime, timedelta


def parse_timelines(file):
    with open(file+'/timelines.jsonl', 'r') as f:
        timeline = json.load(f)

    return timeline


def parse_keywords(file):
    with open(file+'/keywords.json', 'r') as f:
        timeline = json.load(f)

    return timeline


def generate_query(tls, kws, dir):
    queries = []
    for tl in tls:
        print(tl[1])

        q = kws.copy()
        x = input()
        q.append(x)
        while len(x) > 0:
            x = input()
            if len(x) > 0:
                q.append(x)

        queries.append([tl[0], q])

    with open(dir+'/queries.json', 'w') as f:
        json.dump(queries, f, indent=4)
    return


if __name__ == '__main__':
    root = './entities/'

    tl_dirs = [os.path.join(root, name) for name in os.listdir(root) if
                  os.path.isdir(os.path.join(root, name))]
    for tl in tl_dirs:
        print(tl)
        timelines = parse_timelines(tl)
        keywords = parse_keywords(tl)
        generate_query(timelines, keywords, tl)
