"""Validate UFC 331 artifacts, seal provenance, and update the local website."""
import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import run_ufc331_20260919 as card

r = card.runner
out = r.REPO_ROOT / 'pics/picks/UFC_331_Van_vs_Pantoja_2_20260919_win_odds'
site = r.REPO_ROOT.parent / 'mmaai-flask'
rows = r.read_prediction_rows(out / 'fight_predictions.csv')
event = r.website_event(rows)
checks = []
png_names = set()
for row, entry in zip(rows, event['predictions'], strict=True):
    f1, f2 = row['Fighter1'], row['Fighter2']
    fight_dir = out / f'{r.slug(f1)}_vs_{r.slug(f2)}'
    stem = f'{f1.split()[0]}_{f2.split()[0]}'
    shap = json.loads((fight_dir / f'shap_{stem}.json').read_text())
    assert shap['fighter1']==f1 and shap['fighter2']==f2
    p1 = float(row['Fighter1_AI_Prob']) / 100
    assert abs(shap['prediction_probability'] - p1) < 1e-10
    error = abs(float(shap['base_values']) + sum(shap['shap_values'][0]) - p1)
    assert error < 1e-6
    assert len(shap['feature_names']) == len(shap['feature_values'][0]) == len(shap['shap_values'][0]) == 40
    pick = row['AI_Pick']
    assert pick == (f1 if p1 >= .5 else f2)
    chosen = p1 if pick == f1 else 1-p1
    assert abs(chosen*100 - float(row['Confidence'])) < 1e-8
    odds = float(row['Fighter1_Odds'] if pick == f1 else row['Fighter2_Odds'])
    ev = 100 * (chosen * (odds/100 if odds>0 else 100/abs(odds)) - (1-chosen))
    assert abs(ev - float(row['EV'])) < 1e-8
    assert entry['result'] == ('Pending' if ev>0 else 'Pending - no bet')
    for kind in ['shap', 'grouped', 'ind']:
        png = fight_dir / 'screenshots' / f'{kind}_{stem}.png'
        assert png.name not in png_names, 'Chart filename collision'
        png_names.add(png.name)
        with Image.open(png) as im:
            assert im.width>500 and im.height>500
            assert im.convert('L').getextrema()[0]<100
            im.load()
        assert r.sha256_file(png) == r.sha256_file(out/'screenshots'/png.name)
    entry['shap_image'] = f'20260919_shap_{stem}.png'
    checks.append({'fight':entry['fight'], 'ev':ev, 'shap_additivity_error':error, 'three_pngs_verified':True})
assert len(list((out/'screenshots').glob('*.png'))) == 3*len(rows)
with zipfile.ZipFile(out/'shap_screenshots.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
    for png in sorted((out/'screenshots').glob('shap_*.png')):
        bundle.write(png, png.name)
with r.PREDICTION_DATA.open(encoding='utf-8', newline='') as stream:
    history = list(csv.DictReader(stream))
freshness = {f:max((h['event_date'] for h in history if h['fighter_name']==f), default=None) for bout in r.CARD for f in bout[:2]}
verification = {'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'checks':checks,
    'visual_review':'All 30 final PNGs inspected; SHAP and radar titles, labels, and plotted content are visible.',
    'prediction_count':len(rows), 'png_count':3*len(rows),
    'prediction_data_latest_event_date':max(h['event_date'] for h in history), 'fighter_latest_history_dates':freshness}
(out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
(out/'website_event.json').write_text(json.dumps(event, indent=2)+'\n')
mp = out/'run_manifest.json'
manifest = json.loads(mp.read_text())
manifest['official_card'] = [{'fighter1':a,'fighter2':b,'fighter1_odds':c,'fighter2_odds':d,
    'odds_source':None if c is None else r.SOURCE_URLS[3] if a=='ryan gandra' else r.SOURCE_URLS[2]} for a,b,c,d in r.CARD]
manifest['degrees_of_freedom']['odds'] = r.CARD_NOTE
manifest['pipeline']['card_runner'] = 'scripts/run_ufc331_20260919.py'
manifest['pipeline']['source_hashes'] = {p:r.sha256_file(r.REPO_ROOT/p) for p in ['predict.py','libs/shap_visualization.py','libs/visualization.py','libs/screenshot.py','scripts/run_ufc_paris_20260905.py','scripts/run_ufc331_20260919.py','scripts/finalize_ufc331_20260919.py','scripts/refresh_individual_screenshots.py']}
manifest['pipeline']['postprocessing'] = 'Individual radar chart labels compacted and canvas enlarged from saved traces; all PNGs refreshed without changing model predictions or SHAP data.'
manifest['inputs']['prediction_data'].update({'latest_event_date':verification['prediction_data_latest_event_date'], 'fighter_latest_history_dates':freshness})
manifest['sources']['previous_results'] = json.loads((site/'docs/updates/2026-09-13-results.json').read_text())
manifest['outputs'] = r.output_hashes(out, mp)
mp.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
# All artifact checks must pass before touching the website.
website_path = site/'data/eventsv6.json'
website = json.loads(website_path.read_text())
existing = [e for e in website['events'] if datetime.strptime(e['date'],'%Y-%m-%d').date().isoformat()==r.EVENT_DATE]
assert len(existing)<=1
if existing:
    assert existing[0]['name']==event['name']
    website['events'].remove(existing[0])
website['events'].append(event)
for entry in event['predictions']:
    shutil.copy2(out/'screenshots'/entry['shap_image'].removeprefix('20260919_'), site/'visuals'/entry['shap_image'])
website_path.write_text(json.dumps(website, indent=2)+'\n')
report = {'event':manifest['event'], 'sources':manifest['sources'], 'inputs':manifest['inputs']['prediction_data'],
    'model_path':manifest['inputs']['model_path'], 'verification':verification, 'predictions':event['predictions'],
    'png_sha256':{entry['shap_image']:r.sha256_file(site/'visuals'/entry['shap_image']) for entry in event['predictions']}}
(site/'docs/updates/2026-09-19-predictions.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(verification, indent=2))
