"""Frozen UFC 331 card and prices, verified September 13, 2026."""
import run_ufc_paris_20260905 as runner

runner.EVENT_NAME = 'UFC 331: Van vs Pantoja 2'
runner.EVENT_DATE = '2026-09-19'
runner.SOURCE_RETRIEVED_AT_UTC = '2026-09-13T21:00:00Z'
runner.SOURCE_URLS = [
 'https://www.ufc.com/events?language_content_entity=en',
 'https://www.ufc.com.br/news/ufc-331-card-completo-horario-onde-assistir-ao-vivo',
 'https://sbaodds.com/mma-odds/',
 'https://www.sbgglobal.eu/sports-betting-live-lines/fighting/ufc/',
 'https://www.covers.com/sport/mma/ufc/odds',
]
runner.CARD_NOTE = ('Official 13-bout September 19 card. Odds snapshot retrieved September 13 around 21:00 UTC: '
 'SBA Odds except Gandra/Diaz from SBGGlobal (-787/+415). Current Aswell/Yoo odds unavailable across checked pages; '
 'do not substitute September 2 opening lines. Dataset aliases: Doo Ho Choi = dooho choi; '
 'Joo Sang Yoo = joosang yoo; Michael Aswell = michael aswell jr.; Osman Diaz = ozzy diaz. '
 'Main and co-main scheduled for five rounds. Existing production model and feature policy retained.')
runner.CARD = [
 ('joshua van','alexandre pantoja',-115,-101),
 ('arman tsarukyan','mauricio ruffy',-325,265),
 ('patricio pitbull','dooho choi',216,-260),
 ('renato moicano','brian ortega',-200,170),
 ('alonzo menifield','iwo baraniewski',224,-270),
 ('gable steveson','sean sharaf',-1500,891),
 ('marlon vera','charles jourdain',215,-255),
 ('tai tuivasa','robelis despaigne',475,-650),
 ('michael aswell jr.','joosang yoo',None,None),
 ('ryan gandra','ozzy diaz',-787,415),
 ('edmen shahbazyan','brunno ferreira',-150,128),
 ("casey o'neill",'eduarda moura',-195,165),
 ('giga chikadze','joanderson brito',273,-335),
]
if __name__ == '__main__':
 raise SystemExit(runner.main())
