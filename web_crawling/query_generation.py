import os
import json
from datetime import datetime, timedelta


def parse_timelines(file):
    # get timelines from file
    with open(file+'/timelines.jsonl', 'r') as f:
        timeline = json.load(f)

    return timeline


def parse_keywords(file):
    # get keyword from file
    with open(file+'/keywords.json', 'r') as f:
        timeline = json.load(f)

    return timeline


def generate_query(tls, kws, dir):
    # generate query, the format of query is [time, [keyword1, keyword2, ..., keyword_n]]
    queries = []
    for tl in tls:
        print(tl[1])

        q = kws.copy()
        # type one keyword each time, smash enter to skip
        x = input()
        q.append(x)
        while len(x) > 0:
            x = input()
            if len(x) > 0:
                q.append(x)

        queries.append([tl[0], q])

    # new query will be stored in queries.json file
    with open(dir+'/queries.json', 'w') as f:
        json.dump(queries, f, indent=4)
    return


if __name__ == '__main__':
    root = './entities/'

    # visit the directory, yo can change it to any directory as you want
    print(root)
    timelines = parse_timelines(root)
    keywords = parse_keywords(root)
    generate_query(timelines, keywords, root)
