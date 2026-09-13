"""Refresh SHAP HTML/PNG from persisted values without recomputing explanations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from libs.shap_visualization import ShapVisualizer
from libs.screenshot import take_screenshots

parser=argparse.ArgumentParser()
parser.add_argument('event_dir',type=Path)
parser.add_argument('--fighter1', help='Refresh only this fighter1 instead of every saved chart.')
args=parser.parse_args()
for source in sorted(args.event_dir.glob('*/shap_*.json')):
 record=json.loads(source.read_text())
 if args.fighter1 and record['fighter1'] != args.fighter1:
  continue
 viz=ShapVisualizer.__new__(ShapVisualizer)
 viz.feature_display_names={}
 viz.output_dir=str(source.parent)
 data={'shap_values':np.array(record['shap_values']), 'feature_names':record['feature_names'], 'expected_value':record['base_values']}
 fig=viz.create_force_plot(data,fighter1_name=record['fighter1'],fighter2_name=record['fighter2'],win_prob=record['prediction_probability'])
 viz.save_plot(fig,record['fighter1'],record['fighter2'])
 take_screenshots(str(source.parent))
 print('Refreshed',source,flush=True)
