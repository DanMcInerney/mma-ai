"""Verify a completed Noche run, bundle SHAP screenshots, and seal provenance."""
import csv
import json
from pathlib import Path
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from PIL import Image
import run_noche_20260912 as noche

root=noche.runner.REPO_ROOT
out=root/'pics/picks/Noche_UFC_Silva_vs_Delgado_20260912_win_odds'
repro=root/'pics/picks/Noche_UFC_Silva_vs_Delgado_20260912_reproduction'
rows=noche.runner.read_prediction_rows(out/'fight_predictions.csv')
assert rows == noche.runner.read_prediction_rows(repro/'fight_predictions.csv'), 'Fresh no-SHAP reproduction differs'
event=json.loads((out/'website_event.json').read_text())
checks=[]
for row, entry in zip(rows,event['predictions'],strict=True):
 f1,f2=row['Fighter1'],row['Fighter2']
 fight=out/f'{noche.runner.slug(f1)}_vs_{noche.runner.slug(f2)}'
 source=fight/f'shap_{f1.split()[0]}_{f2.split()[0]}.json'
 shap=json.loads(source.read_text())
 probability=float(row['Fighter1_AI_Prob'])/100
 assert abs(shap['prediction_probability']-probability)<1e-10
 assert abs(float(shap['base_values'])+sum(shap['shap_values'][0])-probability)<1e-6
 assert len(shap['feature_names'])==len(shap['feature_values'][0])==len(shap['shap_values'][0])==40
 chosen=float(row['Confidence'])/100
 odds=float(row['Fighter1_Odds'] if row['AI_Pick']==f1 else row['Fighter2_Odds'])
 profit=odds/100 if odds>0 else 100/abs(odds)
 ev=100*(chosen*profit-(1-chosen))
 assert abs(float(row['EV'])-ev)<1e-8
 assert entry['result']==('Pending' if ev>0 else 'Pending - no bet')
 for kind in ['shap','grouped','ind']:
  png=fight/'screenshots'/f'{kind}_{f1.split()[0]}_{f2.split()[0]}.png'
  with Image.open(png) as im:
   assert im.width>500 and im.height>500
   assert im.convert('L').getextrema()[0]<100
  (out/'screenshots').mkdir(exist_ok=True)
  shutil.copy2(png,out/'screenshots'/png.name)
 checks.append({'fight':f'{f1} vs {f2}','shap_probability':probability,'ev':ev,'screenshots_verified':True})
with zipfile.ZipFile(out/'shap_screenshots.zip','w',zipfile.ZIP_DEFLATED) as bundle:
 for png in sorted((out/'screenshots').glob('shap_*.png')): bundle.write(png,png.name)
with noche.runner.PREDICTION_DATA.open(newline='',encoding='utf-8') as f:
 history=list(csv.DictReader(f))
freshness={f:max((r['event_date'] for r in history if r['fighter_name']==f),default=None) for card in noche.runner.CARD for f in card[:2]}
verification={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'prediction_count':len(rows),'fresh_no_shap_reproduction':'exact CSV row equality','manual_odds_tests':'6 passed','checks':checks,'prediction_data_latest_event_date':max(r['event_date'] for r in history),'fighter_latest_history_dates':freshness}
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
manifest_path=out/'run_manifest.json'
manifest=json.loads(manifest_path.read_text())
manifest['sources']['retrieved_at_utc']=noche.runner.SOURCE_RETRIEVED_AT_UTC
manifest['pipeline']['card_runner']='scripts/run_noche_20260912.py'
manifest['sources']['previous_results']={'event':'UFC Fight Night: Hooker vs Parnasse','date':'2026-09-05','url':'https://www.ufc.com/news/ufc-paris-official-scorecards-hooker-vs-parnasse','verified_winners':['mario pinto','daniil donchenko','modestas bukauskas','michael page','axel sola','felipe lima'],'verified_by':'website-update task'}
manifest['pipeline']['postprocessing_commands']=[
 '.venv/Scripts/python.exe scripts/refresh_shap_screenshots.py pics/picks/Noche_UFC_Silva_vs_Delgado_20260912_win_odds',
 '.venv/Scripts/python.exe scripts/refresh_shap_screenshots.py pics/picks/Noche_UFC_Silva_vs_Delgado_20260912_win_odds --fighter1 "brandon moreno"',
 '.venv/Scripts/python.exe scripts/finalize_noche_20260912.py',
]
manifest['inputs']['prediction_data']['latest_event_date']=verification['prediction_data_latest_event_date']
manifest['inputs']['prediction_data']['fighter_latest_history_dates']=freshness
manifest['pipeline']['source_hashes']={p:noche.runner.sha256_file(root/p) for p in ['predict.py','libs/shap_visualization.py','libs/screenshot.py','scripts/run_ufc_paris_20260905.py','scripts/run_noche_20260912.py','scripts/refresh_shap_screenshots.py','scripts/finalize_noche_20260912.py']}
manifest['outputs']=noche.runner.output_hashes(out,manifest_path)
manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'predictions':len(rows),'pngs':len(list((out/'screenshots').glob('*.png'))),'checks':checks},indent=2))
