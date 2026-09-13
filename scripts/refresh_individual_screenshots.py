"""Refresh radar layout from saved Plotly traces without rerunning inference."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.graph_objects as go
from libs.visualization import compact_feature_label
from libs.screenshot import take_screenshots

out = Path(sys.argv[1]).resolve()
decoder = json.JSONDecoder()
for folder in sorted(out.iterdir()):
    if len(sys.argv)>2 and folder.name != sys.argv[2]:
        continue
    if not folder.is_dir() or not list(folder.glob('shap_*.json')):
        continue
    for path in folder.glob('ind_*.html'):
        html = path.read_text(encoding='utf-8')
        remaining = html[html.rindex('Plotly.newPlot(')+len('Plotly.newPlot('):].lstrip()
        args = []
        for _ in range(3):
            value, end = decoder.raw_decode(remaining)
            args.append(value)
            remaining = remaining[end:].lstrip().removeprefix(',').lstrip()
        fig = go.Figure(data=args[1], layout=args[2])
        for trace in fig.data:
            trace.theta = [compact_feature_label(t) if '_' in t else t for t in trace.theta]
        fig.update_layout(width=1800, height=1800, margin=dict(l=320,r=320,t=250,b=200),
            legend=dict(orientation='h',x=.5,xanchor='center',y=1.08),polar_angularaxis_tickfont_size=11)
        fig.write_html(path)
    take_screenshots(str(folder))
    for png in (folder/'screenshots').glob('*.png'):
        shutil.copy2(png, out/'screenshots'/png.name)
