import requests
import json
from bs4 import BeautifulSoup
from datetime import datetime

now = datetime.now()

def scrape_value(currency='bitcoin'):
    r = requests.get('https://www.investing.com/crypto/' + currency, headers={'User-Agent':'Mozilla/5.0'})
    c = r.content
    soup = BeautifulSoup(c, 'html.parser')

    value_raw = soup.findAll('span', {'id': 'last_last'})
    float(value_raw[0].text.replace(',',''))
    
    value_net = float(value_raw[0].text.replace(',',''))
    return value_net

currentInfo = {
    'timeActivation': 'hello',
    'currentValue': scrape_value()
}

print(currentInfo)

requests.post('http://localhost:3000/post', json=currentInfo)
