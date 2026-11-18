# Kami

Give it a research paper or a long article and it shows you the text itself with the important parts highlighted. Inputs:

- a PDF link, an arXiv link (`/abs/`, `/pdf/`, `/html/` or a bare id such as `1706.03762`; the PDF is always fetched), or an OpenReview forum link
- a blog post or article URL (personal blogs, Substack, docs pages, Paul Graham style essays); the main article text is extracted and shown in a clean reading view
- an uploaded PDF, a photo or screenshot of a page, or a saved `.html` file (drag and drop, "choose a file", or paste an image)

What gets marked:

| Color | Meaning | Note on hover |
| --- | --- | --- |
| Amber | Key claims of the paper | The claim in plain English, plus what evidence backs it |
| Yellow | Important sentences in a section | Why it matters |
| Violet | Complex arguments | The same idea restated simply |
| Teal | Definitions of new terms | Short definition |
| Section labels | STUDY / READ / SKIM / SKIP next to each heading, with difficulty D1 to D5 | Why, prerequisites, signals |

PDFs and images are shown as page images with the highlights drawn on top. Web articles are shown as text in a reading column with the highlights as `<mark>` spans and the section labels next to each heading.

Hover a highlight to see its note, click to pin it. The legend chips toggle each highlight type. The side panel (button "Reading plan") holds the ranked reading plan with self-check questions per section, the list of main claims (click to jump), and the prerequisites to learn first.

## The interface

- Home: one box for a link, a drop area that also opens a file picker, three examples (the bundled Attention paper works offline; the arXiv and blog examples need internet), a key to the highlight colors, and a "Recently read" list from the local cache.
- Reader: the wordmark and "New paper" go back home and reset state; the browser back button works too. The legend bar toggles each highlight type. The side panel has the reading plan, the main claims (click to jump) and prerequisites; under 1100px wide it becomes a drawer opened with "Reading plan".
- Fonts are bundled in `static/fonts` (Newsreader for reading, IBM Plex Sans for the interface, IBM Plex Mono for code; all SIL OFL), so nothing loads from a CDN.

## How it works

```
input (URL / PDF / image / web page)
  -> app.py       arXiv and OpenReview links are rewritten to their PDF; other links are fetched
                  with a browser User-Agent; a page that declares citation_pdf_url gets its PDF
  -> parse.py     PyMuPDF text with exact word boxes (PDF) or RapidOCR boxes (image),
                  heading detection, sections, line-end hyphen joining
  -> article.py   web pages: trafilatura (readability-lxml as fallback) extracts the main article,
                  keeps headings, paragraphs, lists, quotes and code; headings become sections;
                  long essays with no headings are cut into parts of about 900 words
  -> signals.py   cheap per-section stats: equation lines, new terms, citations
  -> analyze.py   LLM pass 1: main claims (abstract + intro + conclusion), each with a verbatim quote
                  LLM pass 2: one call per section -> supports_claims, difficulty 1-5,
                  key_points, complex_arguments (+ plain restatement), definitions,
                  prerequisites, self_check
                  LLM pass 3: reading plan, every section bucketed study / read / skim / skip
  -> spans.py     every quote is matched back to exact token ranges (exact, then anchored, then
                  fuzzy match) and turned into per-line rectangles in page coordinates
  -> JSON         pages, sections, claims, plan, highlights[{type, note, quote, rects[]}]
                  (web articles: blocks[] and tokens[] instead of pages, highlights carry token ranges)
frontend          renders each page as an image and draws the rectangles on top, or renders the
                  article text and wraps highlighted token ranges in <mark>
```

Every LLM answer is validated with Pydantic (`pipeline/schemas.py`). A bad answer is retried once with the validation error attached; if a section still fails, that section is marked and the rest of the paper continues. Results are cached per file hash (for web pages, per extracted article text) and provider/model under `data/<sha256[:16]>/`.

Web page errors are reported plainly: HTTP 401/402/403 is explained as a paywall, sign-in or bot block; a page with under 120 words of article text (paywall teaser, JavaScript-only app) is rejected with a suggestion to use the PDF or save the page as PDF.

## Setup

```bash
cd paper-highlighter
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then put ONE API key in .env
./run.sh                    # http://localhost:8765
```

No system packages are needed. OCR uses RapidOCR (ONNX models ship in the pip package), so there is no Tesseract install.

## LLM provider

Set one key. The provider is picked automatically from the first key found, or forced with `LLM_PROVIDER`.

| LLM_PROVIDER | Key env var | Default model (override with LLM_MODEL) |
| --- | --- | --- |
| openai | OPENAI_API_KEY | gpt-4.1-mini |
| anthropic | ANTHROPIC_API_KEY | claude-haiku-4-5 |
| gemini | GEMINI_API_KEY | gemini-2.5-flash |
| xai | XAI_API_KEY | grok-3-mini |
| groq | GROQ_API_KEY | llama-3.3-70b-versatile |
| openrouter | OPENROUTER_API_KEY | openai/gpt-4.1-mini |
| mock | none | offline heuristic demo |

`LLM_BASE_URL` points any OpenAI-compatible provider at another server, for example a local Ollama (`LLM_PROVIDER=openai`, `OPENAI_API_KEY=ollama`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5:7b`).

With no key the app still works end to end using the built-in mock analyzer (`pipeline/mock_llm.py`). It quotes real sentences picked by keyword and math-density heuristics, so highlights render correctly, but the notes are placeholders marked "Demo note". The UI shows a banner in this mode. `GET /api/config` reports which provider is active (never the key itself).

A typical paper is about 20 to 30 LLM calls (1 claims + 1 per section + 1 plan), run 4 at a time.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | /api/analyze | form field `url` (PDF, arXiv, or web page) or `file` (PDF, image, .html), optional `force=true` to ignore cache. Returns `{doc_id, kind}` |
| POST | /api/sample/attention | analyze the bundled sample paper (`samples/attention.pdf`) |
| GET | /api/recent | recently analyzed documents from the cache (title, kind, time) |
| GET | /api/status/{doc_id} | `{status: running/done/error, stage, progress}` |
| GET | /api/doc/{doc_id} | the full analysis JSON |
| GET | /api/doc/{doc_id}/page/{n}.png | rendered page image (PDF and image input only) |
| GET | /api/config | active provider and which key names are present |

The reader also accepts `/?url=<link>` to start immediately and `/?doc=<doc_id>` to reopen a result. You can drag and drop or paste a file anywhere on the page.

## Tests

```bash
.venv/bin/pytest -q tests          # 23 tests, no network: parsing, spans, pipeline, retry, provider HTTP,
                                   # image OCR, web article extraction, paywall rejection, arXiv link
                                   # rewriting, fetch via a local HTTP server, the HTTP API end to end
.venv/bin/python -m playwright install chromium   # or use a system Chrome, see tests/screenshot.py
.venv/bin/python tests/screenshot.py               # with the server running: clicks through home, the
                                   # sample PDF, Home, a blog URL and the plan; writes shot-home.png,
                                   # shot-pdf.png, shot-blog.png and 390px wide shot-*-narrow.png
                                   # (the blog step needs internet)
```

## Files

```
app.py                 FastAPI app: input handling, jobs, cache, page rendering, static files
pipeline/parse.py      PDF and image parsing into words, tokens, sections
pipeline/article.py    web article extraction and conversion to the same document model
pipeline/signals.py    per-section signals
pipeline/llm.py        pluggable provider client
pipeline/mock_llm.py   offline analyzer used when no key is set
pipeline/analyze.py    the pipeline and highlight mapping
pipeline/spans.py      quote to text span to rectangle matching
pipeline/schemas.py    Pydantic models for LLM output
prompts/*.txt          the three prompts
static/                index.html, app.js, style.css, favicon.svg, fonts/
samples/attention.pdf  the sample paper behind the home page example
tests/                 pytest suite, HTML fixtures, screenshot script, sample paper
```

## Known limitations

- Scanned PDFs without a text layer are rejected; upload page images instead (OCR handles one image at a time).
- Heading detection relies on bold or larger numbered headings. Papers with unusual layouts may come out as fewer, larger sections; the analysis still works, the plan is just coarser.
- Footnotes and page footers are intentionally excluded from analysis.
- Text inside figures and most math is not highlighted; equations are counted as a signal only.
- Highlights on PDFs are drawn over page images, so PDF text in the reader is not selectable (article text is).
- Some sites refuse automated requests outright (Medium returns HTTP 403 to this server, and sites behind Cloudflare challenges or hard paywalls do the same). Save the page as PDF and upload it instead.
- Pages that build their content with JavaScript have no text in the HTML and are rejected. Images, figures and tables inside articles are dropped or flattened to text rows.
- Jobs live in memory; if the server restarts mid-analysis, resubmit (finished results stay cached on disk).
