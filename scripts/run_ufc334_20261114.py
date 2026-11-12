"""Run the frozen November 14 UFC 334 card with origin/main's pipeline."""
import json
from datetime import datetime, timezone

import run_ufc_paris_20260905 as runner

runner.EVENT_NAME = 'UFC 334: Gane vs Hokit'
runner.EVENT_DATE = '2026-11-14'
OUT = runner.REPO_ROOT / 'pics/picks/UFC_334_Gane_vs_Hokit_20261114_win_odds'
CONFIG = runner.REPO_ROOT / 'research/20261112-run-inputs.json'
SOURCE_FILES = [
    'predict.py', 'libs/shap_visualization.py', 'libs/visualization.py',
    'libs/screenshot.py', 'libs/modeling/inference_contract.py',
    'scripts/run_ufc_paris_20260905.py', 'scripts/run_ufc334_20261114.py',
    'research/20261112-run-inputs.json',
]
SOURCE_FILES += sorted(p.relative_to(runner.REPO_ROOT).as_posix()
                       for p in (runner.REPO_ROOT / 'libs/feature_store/inference').rglob('*.py'))


def configure():
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    runner.SOURCE_RETRIEVED_AT_UTC = config['retrieved_at_utc']
    runner.SOURCE_URLS = config['source_urls']
    runner.CARD_NOTE = config['card_note']
    runner.CARD = [tuple(bout) for bout in config['card']]
    runner.MODEL_PATH = runner.REPO_ROOT / config['model_path']
    runner.TRAINING_DATA = runner.MODEL_PATH / 'training_data.csv'
    stems = [f"{a.split()[0].replace(chr(39), '')}_{b.split()[0].replace(chr(39), '')}" for a, b, _, _ in runner.CARD]
    assert len(stems) == len(set(stems)), 'Chart name collision'
    assert not any("'" in n for bout in runner.CARD for n in bout[:2]), "main's runner needs apostrophe-free names"
    return config


def main():
    config = configure()
    args = runner.parse_args()
    if args.resume:
        raise ValueError('Resume requires a separately verified matching input snapshot.')
    if args.no_shap or not args.screenshots:
        raise ValueError('Production runs require SHAP and all screenshots.')
    if args.output_dir.resolve() != OUT.resolve() or args.manifest_path.resolve() != (OUT / 'run_manifest.json').resolve():
        raise ValueError('Output paths must match this frozen event wrapper.')
    if OUT.exists() and list(OUT.iterdir()):
        raise FileExistsError(f'Preserve the existing run: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    before = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'event': runner.EVENT_NAME, 'date': runner.EVENT_DATE,
        'configuration': config,
        'source_hashes': {p: runner.sha256_file(runner.REPO_ROOT / p) for p in SOURCE_FILES},
        'prediction_sha256': runner.sha256_file(runner.PREDICTION_DATA),
        'training_sha256': runner.sha256_file(runner.TRAINING_DATA),
        'model_files': runner.model_file_hashes(),
    }
    (OUT / 'input_snapshot.json').write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
    code = runner.main()
    if code:
        return code
    assert before['prediction_sha256'] == runner.sha256_file(runner.PREDICTION_DATA)
    assert before['training_sha256'] == runner.sha256_file(runner.TRAINING_DATA)
    assert before['model_files'] == runner.model_file_hashes()
    assert before['source_hashes'] == {p: runner.sha256_file(runner.REPO_ROOT / p) for p in SOURCE_FILES}
    mp = OUT / 'run_manifest.json'
    manifest = json.loads(mp.read_text())
    manifest['official_card'] = config['card']
    manifest['pipeline']['card_runner'] = 'scripts/run_ufc334_20261114.py'
    manifest['pipeline']['code_ref'] = config['code_ref']
    manifest['pipeline']['source_hashes'] = before['source_hashes']
    manifest['degrees_of_freedom']['odds'] = config['card_note']
    manifest['outputs'] = runner.output_hashes(OUT, mp)
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
