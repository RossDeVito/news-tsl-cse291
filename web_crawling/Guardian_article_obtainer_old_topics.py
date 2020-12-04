from bs4 import BeautifulSoup as BS
import requests
import re
import json
import os
from theguardian import theguardian_content
from datetime import datetime, timedelta
from tqdm import tqdm

guardian_root = 'https://content.guardianapis.com/search?'
div_classes = ['content__article-body from-content-api js-article__body',
               'article-body-commercial-selector css-79elbk article-body-viewer-selector',
               'css-1uuhxtj',
               'css-avj6db',
               'article-body-commercial-selector css-79elbk']


def parse_keyword(file):
    with open(file, 'r') as f:
        kw = json.load(f)
    return kw


def parse_timeline(file):
    dd = []
    with open(file, 'r') as f:
        for l in f:
            timeline = json.loads(l)
            start_date = datetime.strptime(timeline[0][0], '%Y-%m-%d %H:%M:%S')
            end_date = datetime.strptime(timeline[-1][0], '%Y-%m-%d %H:%M:%S')
            if len(dd) == 0:
                dd.append(start_date)
                dd.append(end_date)
            else:
                if start_date < dd[0]:
                    dd[0] = start_date
                if end_date > dd[0]:
                    dd[1] = end_date

    return dd


def obtain_articles(dir, kws, ak, dates):
    start_date = dates[0]
    end_date = dates[1]
    duration = end_date-start_date
    start_date = start_date - timedelta(days=int(duration.days*0.1))
    end_date = end_date + timedelta(days=int(duration.days*0.1))

    query = kws[0]
    for kw in kws[1:]:
        query += ' AND ' + kw

    print(query)

    content = theguardian_content.Content(api=ak, q=query, from_date=str(start_date.date()),
                                          to_date=str(end_date.date()), page=1)

    headers = content.response_headers()
    pages = headers['pages']
    if pages <= 1:
        print(pages)
        return

    out_file = open(dir + '/articles.jsonl', 'w', encoding='UTF-8')
    index = 1

    for i in range(1, pages+1):
        content = theguardian_content.Content(api=ak, q=query, from_date=str(start_date.date()),
                                              to_date=str(end_date.date()), page=i)
        json_content = content.get_content_response()
        try:
            all_results = content.get_results(json_content)
        except:
            continue
        for r in all_results:
            if r['type'] != 'article':
                continue
            text = get_content(r['fields'])
            if len(text) < 10:
                continue
            publish_date = datetime.strptime(r['webPublicationDate'], '%Y-%m-%dT%H:%M:%SZ')
            publish_date.strftime('%Y-%m-%d %H:%M:%S')

            art = {'id': r['id'],
                   'time': str(publish_date),
                   'text': text,
                   'title': r['webTitle']}
            json.dump(art, out_file)
            out_file.write('\n')
            index += 1
            if index > 1000:
                break
        if index > 1000:
            break

    print(index)
    out_file.close()


def get_content(fields):
    soup = BS(fields['body'], 'lxml')

    return soup.get_text()


if __name__ == '__main__':
    api_key = '4ff3ddcd-148a-4cbe-83f9-682aa211c090'
    root = './datasets/'

    tl_dirs = [os.path.join(root, name) for name in os.listdir(root) if
                  os.path.isdir(os.path.join(root, name))]
    for tl in tl_dirs:
        topic = [os.path.join(tl, name) for name in os.listdir(tl) if
                   os.path.isdir(os.path.join(tl, name))]
        for t in topic:
            keywords = parse_keyword(t+'/keywords.json')
            date_duration = parse_timeline(t+'/timelines.jsonl')
            obtain_articles(t, keywords, api_key, date_duration)
