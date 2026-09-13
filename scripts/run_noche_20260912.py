"""Run the September 12 Noche card through the established prediction runner."""
import run_ufc_paris_20260905 as runner

runner.EVENT_NAME = 'Noche UFC: Silva vs Delgado'
runner.EVENT_DATE = '2026-09-12'
runner.SOURCE_RETRIEVED_AT_UTC = '2026-09-12T12:03:00Z'
runner.SOURCE_URLS = [
 'https://www.ufc.com/event/ufc-fight-night-september-12-2026',
 'https://jp.ufc.com/news/fight-fight-preview-noche-ufc-phoenix-silva-delgado-2026',
 'https://sbaodds.com/mma-odds/',
]
runner.CARD_NOTE = 'Jose Miguel Delgado replaces Yair Rodriguez; dataset alias jose delgado. Frozen current SBA Odds prices supplied from retrieval September 12, 2026. No Silva/Rodriguez odds used.'
runner.CARD = [
 ('jean silva','jose delgado',-425,325),
 ('brandon moreno','joseph morales',-105,-115),
 ('tommy mcmillen','marwan rahiki',-151,131),
 ('manon fiorot','alexa grasso',-270,222),
 ('waldo cortes acosta','curtis blaydes',-190,165),
 ('david martinez','dan ige',-460,358),
 ('tim elliott','edgar chairez',183,-215),
 ('ignacio bahamondes','muslim salikhov',-525,415),
 ('yousri belgaroui','djorden santos',-750,534),
 ('drakkar klose','tommy gantt',335,-435),
 ('rafa garcia','rongzhu',140,-160),
 ('sean king iii','jessie rosas',-205,177),
 ('jj aldrich','regina tarin',234,-285),
]
if __name__ == '__main__':
 raise SystemExit(runner.main())
