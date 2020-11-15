from bs4 import BeautifulSoup as BS
import requests
import re
import json

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


def get_content(soup):
    result = dict()
    content = soup.find_all('a')
    for c in content:
        href = root + c.get('href')
        title = c.get_text()
        timeline = get_timeline(href)
        if len(timeline) < 5:
            continue
        result[title] = timeline
    return result


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
        tl.append([date, text[1]])

    return tl


def convert_date(str):
    s = re.split(' |, ', str)
    if len(s) != 3 or s[0] not in months:
        return -1
    return s[2] + '-' + months[s[0]] + '-' + s[1] + ' ' + '00:00:00'


if __name__ == '__main__':
    output = dict()
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
        output[h_text] = get_content(c)

    with open('cnn_ff_tls.json', 'w', encoding='UTF-8') as outfile:
        json.dump(output, outfile, indent=4)
