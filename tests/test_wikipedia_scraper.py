from bs4 import BeautifulSoup

from libs.wikipedia_scraper import WikiTableScraper


def scraper_for(html, section_id="Scheduled_events"):
    scraper = WikiTableScraper.__new__(WikiTableScraper)
    scraper.id = {"id": section_id}
    scraper.soup = BeautifulSoup(html, "html.parser")
    scraper.base_url = "https://en.wikipedia.org"
    scraper.table = scraper.get_table()
    return scraper


def test_get_table_finds_next_table_when_section_id_is_on_heading():
    scraper = scraper_for(
        """
        <h2 id="Scheduled_events">Scheduled events</h2>
        <p>Introductory text between the heading and table.</p>
        <table><tbody><tr><td>UFC 999</td></tr></tbody></table>
        """
    )

    assert scraper.table is not None
    assert scraper.get_table_column(1) == ["UFC 999"]


def test_get_table_links_resolves_relative_links_and_keeps_absolute_links():
    scraper = scraper_for(
        """
        <table id="Scheduled_events">
          <tbody><tr>
            <td>
              <a href="/wiki/UFC_999">Root relative</a>
              <a href="//upload.wikimedia.org/poster.jpg">Protocol relative</a>
              <a href="https://example.com/event">Absolute</a>
            </td>
          </tr></tbody>
        </table>
        """
    )

    assert scraper.get_table_links(1) == [
        "https://en.wikipedia.org/wiki/UFC_999",
        "https://upload.wikimedia.org/poster.jpg",
        "https://example.com/event",
    ]
