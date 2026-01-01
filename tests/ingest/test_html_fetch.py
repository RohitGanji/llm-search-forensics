from src.ingest.html_fetch import html_to_sections

def test_html_to_sections_heading_paths():
    html = """
    <html><body>
      <h1>Top</h1>
      <p>Intro text here that is long enough to keep.</p>
      <h2>Sub</h2>
      <p>More details that are also long enough to keep as section text.</p>
      <h3>Deep</h3>
      <ul><li>Bullet point with enough content to pass the length filter.</li></ul>
    </body></html>
    """
    secs = html_to_sections(html)
    assert len(secs) >= 2
    assert any("Top" in s.header_path for s in secs)
    assert any("Top > Sub" in s.header_path for s in secs)
