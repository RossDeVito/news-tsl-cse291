from bs4 import BeautifulSoup as BS
import requests
import re
import json
import os
import string
from datetime import datetime

root = 'https://www.cnn.com'
months = {'January': '1',
          'February': '2',
          'March': '3',
          'April': '4',
          'May': '5',
          'June': '6',
          'July': '7',
          'August': '8',
          'September': '9',
          'October': '10',
          'November': '11',
          'December': '12'}
bot_date = datetime.strptime('1998-1-1', '%Y-%m-%d')


def get_content(soup, root_dir):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    content = soup.find_all('a')
    for c in content:
        href = 'https://www.cnn.com' + c.get('href')
        title = c.get_text()
        title_nopuncts = title.translate(translator)
        title_nopuncts = title_nopuncts.replace(' ', '_')
        dir = root_dir+title_nopuncts+'/'
        timeline = get_timeline(href)
        if len(timeline) < 5:
            continue

        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir+'keywords.json', 'w', encoding='UTF-8') as k_file:
            json.dump([title], k_file)

        with open(dir+'timelines.jsonl', 'w', encoding='UTF-8') as outfile:
            json.dump(timeline, outfile, indent=4)


def get_timeline(url):
    tl = []
    page = requests.get(url)
    while page.status_code != 200:
        page = requests.get(url)

    soup = BS(page.content, 'lxml')
    lines = soup.find_all('div', class_='zn-body__paragraph')
    for l in lines:
        text = l.get_text().split(' - ')
        if len(text) < 2:
            continue
        elif len(text) > 2:
            text = [text[0], ''.join(text[1:])]
        date = convert_date(text[0])
        if date == -1:
            continue
        try:
            format_date = datetime.strptime(date, '%Y-%m-%d')
            if format_date < bot_date:
                continue
            tl.append([str(format_date), text[1]])
        except:
            continue
    return tl


def convert_date(str):
    s = re.split(' |, ', str.strip())
    if len(s) != 3 or s[0] not in months:
        return -1

    if '-' in s[1] or '-' in s[0] or '-' in s[2]:
        return -1

    return s[2] + '-' + months[s[0]] + '-' + s[1]


if __name__ == '__main__':
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    root_dir = './Dataset/'
    root = 'https://www.cnn.com'
    start_url = root + '/specials/world/fast-facts'
    r = requests.get(start_url)
    while r.status_code != 200:
        r = requests.get(start_url)

    soup = BS(r.content, 'lxml')

    header = soup.find_all('h2', class_='zn-header__text')
    container = soup.find_all('div', class_='zn__containers')

    for h, c in zip(header, container):
        h_text = h.getText()
        h_text = h_text[:-1]
        print(h_text)
        if h_text == 'People': continue
        h_nopuncts = h_text.translate(translator)
        h_path = h_nopuncts.replace(' ', '_')
        curr_dir = root_dir+h_path+'/'
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        get_content(c, curr_dir)
