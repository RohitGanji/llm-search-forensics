import json
from src.ingest.html_fetch import html_to_sections
from src.ingest.chunk import chunk_text


def test_html_to_sections_basic_structure():
    html = """
    <html><body>
      <h1>Top</h1>
      <p>This is an intro paragraph that is definitely long enough to keep.</p>
      <h2>Sub</h2>
      <p>This section has enough content to be included and parsed properly.</p>
      <h3>Deep</h3>
      <ul>
        <li>This bullet point is long enough to be included in the extracted text content.</li>
      </ul>
    </body></html>
    """
    secs = html_to_sections(html)
    assert len(secs) >= 2
    assert any("Top" in s.header_path for s in secs)
    assert any("Top > Sub" in s.header_path for s in secs)

def test_chunk_text_overlap_contract():
    text = "a" * 5000
    chunks = chunk_text(text, max_chars=2400, overlap_chars=300)
    assert len(chunks) >= 2

    s0, e0, c0 = chunks[0]
    s1, e1, c1 = chunks[1]
    assert s0 == 0
    assert s1 == max(0, e0 - 300)
    assert c0[-300:] == c1[:300]

def test_chunk_text_param_validation():
    try:
        chunk_text("abc", max_chars=10, overlap_chars=10)
        assert False, "Expected ValueError when overlap_chars >= max_chars"
    except ValueError:
        pass
