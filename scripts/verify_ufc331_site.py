"""Check the rendered event, retained results, and exact served chart bytes."""
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

site = Path(__file__).resolve().parents[2]/'mmaai-flask'
base = sys.argv[1]
data = json.loads((site/'data/eventsv6.json').read_text())
event = next(e for e in data['events'] if e['date']=='2026-9-19')
html = urlopen(base, timeout=30).read().decode()
soup = BeautifulSoup(html, 'html.parser')
sections = soup.select('.events-container > .event-section')
latest = sections[0]
assert event['name'] in latest.get_text()
assert '2026-9-19' in latest.get_text()
assert 'Predictions cover' not in soup.get_text()
assert not any(a.get_text(strip=True)=='SHAP chart' for a in soup.select('a'))
assert len(latest.select('td.fight-cell')) == len(event['predictions'])
for tr, entry in zip(latest.select('tbody tr'), event['predictions'], strict=True):
    cells = tr.find_all('td')
    assert cells[0].get_text(strip=True)==entry['fight']
    assert cells[1].get_text(strip=True)==entry['prediction']
    assert cells[-1].get_text(strip=True)==entry['result']
    assert cells[0].select_one('img')['src']=='/visuals/'+entry['shap_image']
prior = next(e for e in data['events'] if e['date']=='2026-9-12')
assert prior['name'] in sections[1].get_text()
for tr, entry in zip(sections[1].select('tbody tr'), prior['predictions'], strict=True):
    assert tr.find_all('td')[-1].get_text(strip=True)==entry['result']
def check_png(entry):
    received = urlopen(urljoin(base,'visuals/'+entry['shap_image']),timeout=30).read()
    assert received == (site/'visuals'/entry['shap_image']).read_bytes()
    return entry['shap_image']
with ThreadPoolExecutor(max_workers=6) as pool:
    checked = list(pool.map(check_png,event['predictions']))
print(json.dumps({'url':base,'event':event['name'],'predictions':len(event['predictions']),
    'previous_results_verified':len(prior['predictions']),'exact_pngs_verified':checked},indent=2))
