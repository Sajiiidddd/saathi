# Knowledge base layout — data/

    classics/       public-domain wisdom texts: raw .txt (fetched, gitignored)
                    plus curated .md passage selections (tracked — the corpus)
    gov_health/     US government mental-health education (public domain)
    nhs_who/        NHS self-help techniques + WHO stress guidance
    open_textbooks/ NOBA psychology chapters (CC BY-NC-SA)
    quotes/         quotes.json — emotion-tagged closing quotes (not RAG content)
    helplines.json  crisis-handoff directory (not RAG content)

Every retrievable `.md` carries YAML front-matter: `title`, `source`, `url`,
`licence`, `retrieved`, `type` (technique | education | wisdom), `topics[]`.
The ingestion pipeline (`saathi.kb.ingest`):

1. parses front-matter → chunk metadata (`source` powers spoken citations)
2. chunks 300–800 chars, heading-aware, never splitting a list mid-sequence
3. embeds locally → exact-search vector index; text → BM25
4. `topics[]` feed the emotion-crosswalk retrieval lane

To (re)fetch the classics raw texts: `python3 scripts/fetch_classics.py`.
Curated passages are selected by hand from those files — selective, short, and
tagged `type: wisdom` so retrieval can prefer techniques for how-to questions.
Corpus guideline: roughly 60% technique/education to 40% wisdom, because long
philosophical chunks otherwise drown the material that answers "help me sleep".

`helplines.json` is loaded by the safety gate and rendered on the landing page.
It is deliberately outside the retrieval index. Verify every number before any
public deployment.

Licence obligations per source are in `LICENSES.md`.
