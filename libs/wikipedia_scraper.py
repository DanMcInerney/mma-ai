import requests
from bs4 import BeautifulSoup
from lxml import etree
import pandas as pd


class WikiTableScraper:
    def __init__(self, url, id):
        self.url = url
        self.id = id
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.soup = self.get_soup(requests.get(self.url, headers=headers))
        self.base_url = self.get_base_url()
        self.table = self.get_table()

    # def run(self, url, tid):
    #     r = requests.get(url)
    #     soup = self.get_soup(r)
    #     events_df = self.get_table_by_id(soup, tid)
    #     return events_df


    def get_base_url(self):
        split = self.url.split('://', 1)
        proto = split[0] + '://'
        domain = split[1].split('/', 1)[0]
        return proto + domain

    def get_soup(self, r):
        soup = BeautifulSoup(r.text, "html.parser")
        return soup

    def get_table(self):
        table = self.soup.find('table', self.id)
        return table

    def get_table_by_id(self):
        pe_df = pd.read_html(str(self.table))[0]
        return pe_df

    def get_table_links(self, column):
        links = []
        for tag in self.table.select(f"td:nth-of-type({column}) a"):
            links.append(self.base_url + tag['href'])
        return links

    def get_table_column(self, column):
        data = []
        for tag in self.table.select(f"td:nth-of-type({column})"):
            data.append(tag.text.strip())
        return data

    def wikipedia_name_conversion(self, wiki_names):
        # Wiki name: Ufcstats name
        name_conversion = {
            "Zachary Reese": "Zach Reese",
            "Lee Chang-ho": "ChangHo Lee",
            "Montserrat Ruiz": "Montserrat Conejo Ruiz",
            "Ko Seok-hyun": "Seokhyeon Ko",
            "Jose Miguel Delgado": "Jose Delgado",
            "Osman Diaz": "Ozzy Diaz",
            "Aaron Brink": "Aaron Brink Jr.",
            "Ateba Abega Gautier": "Ateba Gautier",
            "You Su-young": "SuYoung You",
            'Da Un Jung': 'Da Woon Jung',
            'Jung Da-woon': 'Da Woon Jung',
            'C.J. Vergara': 'CJ Vergara',
            'Alexander Romanov': 'Alexandr Romanov',
            'Kang Kyung-Ho': 'Kyung Ho Kang',
            'Khalil Rountree Jr': 'Khalil Rountree Jr.',
            'Sergey Spivak': 'Serghei Spivac',
            'Jung Chan-Sung': 'Chan Sung Jung',
            'Park Jun-yong': 'JunYong Park',
            'Park Jun-Yong': 'JunYong Park',
            'Jesus Santos Aguilar': 'Jesus Aguilar',
            'Abusupiyan Magomedov': 'Abus Magomedov',
            'Benoit Saint-Denis': 'Benoit Saint Denis',
            'Elves Brenner': 'Elves Brener',
            'Philip Rowe': 'Phil Rowe',
            'Ian Machado Garry': 'Ian Garry',
            'Dennis Tiuliulin': 'Denis Tiuliulin',
            'Carl Deaton III': 'Carl Deaton',
            'Ode\' Osbourne': 'Ode Osbourne',
            'Lee Jeong-yeong': 'JeongYeong Lee',
            "Choi Doo-ho": 'Dooho Choi',
            'Seung Woo Choi': "SeungWoo Choi",
            'Choi Seung-woo': "SeungWoo Choi",
            "Sharabutdin Magomedov": "Shara Magomedov",
            "Daniel Argueta": "Dan Argueta",
            "Hayisaer Maheshate": "Maheshate Hayisaer",
            "Bruno Gustavo da Silva": "Bruno Silva",
            "Carlos Diego Ferreira": "Diego Ferreira",
            "Park Hyun-sung": "HyunSung Park",
            "Jose Daniel Medina": "Jose Medina",
            "Ateba Abega Gautier": "Ateba Gautier",
            "Lee Chang-Ho": "Changho Lee"
        }

        converted_names = []
        for wiki_name in wiki_names:
            if wiki_name in name_conversion.keys():
                converted_names.append(name_conversion[wiki_name].lower())
            else:
                if '(C)' in wiki_name:
                    wiki_name = wiki_name.replace('(C)', '').strip()
                if '(IC)' in wiki_name:
                    wiki_name = wiki_name.replace('(IC)', '').strip()
                converted_names.append(wiki_name.lower())

        return converted_names

# url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'
# id = {'id': 'Scheduled_events'}
# ws = WikiTableScraper(url, id)
# #events_df = ws.get_table_by_id()
# columns = '1'
# event_links = ws.get_table_links(column)






