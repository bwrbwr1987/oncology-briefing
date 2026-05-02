#!/usr/bin/env python3
"""
Oncology Daily Briefing System
PubMed → Claude API → Supabase (web dashboard reads from here)
"""

import os
import json
import time
import hashlib
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from anthropic import Anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_KEY"]

PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "oncology-bot@example.com")

SEARCH_QUERIES = [
    "oncology[MeSH] AND clinical trial[pt] AND last 1 days[dp]",
    "targeted therapy cancer AND last 1 days[dp]",
    "immunotherapy solid tumor AND last 1 days[dp]",
    "HER2 breast cancer AND last 1 days[dp]",
    "colorectal cancer treatment AND last 1 days[dp]",
]

MAX_ARTICLES_PER_QUERY = 5
MAX_TOTAL_ARTICLES     = 15
BKK_TZ = timezone(timedelta(hours=7))


# ── PubMed ────────────────────────────────────────────────────────────────────

def pubmed_search(query: str, max_results: int = MAX_ARTICLES_PER_QUERY) -> list[str]:
    params = {
        "db": "pubmed", "term": query, "retmax": max_results,
        "retmode": "json", "tool": "oncology-briefing",
        "email": PUBMED_EMAIL, "sort": "pub+date",
    }
    r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def pubmed_fetch(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
        "tool": "oncology-briefing", "email": PUBMED_EMAIL,
    }
    r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=20)
    r.raise_for_status()

    articles = []
    root = ET.fromstring(r.text)
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid     = article.findtext(".//PMID") or "N/A"
            title_el = article.find(".//ArticleTitle")
            title    = "".join(title_el.itertext()) if title_el is not None else "No title"
            abs_el   = article.find(".//AbstractText")
            abstract = "".join(abs_el.itertext()) if abs_el is not None else "No abstract available."
            authors  = []
            for a in article.findall(".//Author")[:3]:
                last = a.findtext("LastName", "")
                if last:
                    authors.append(last + " " + a.findtext("ForeName", ""))
            author_str = ", ".join(authors) + (" et al." if len(authors) == 3 else "")
            journal  = article.findtext(".//Journal/Title") or article.findtext(".//MedlineTA") or "Unknown Journal"
            year     = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate", "")[:4]
            articles.append({
                "pmid": pmid, "title": title.strip(), "authors": author_str,
                "journal": journal, "year": year, "abstract": abstract[:1200],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        except Exception as e:
            print(f"  ⚠ Parse error: {e}")
    return articles


def collect_unique_articles() -> list[dict]:
    seen: set[str] = set()
    all_articles: list[dict] = []
    for query in SEARCH_QUERIES:
        print(f"  🔍 {query[:60]}…")
        pmids = pubmed_search(query)
        new_pmids = [p for p in pmids if p not in seen]
        if not new_pmids:
            continue
        seen.update(new_pmids)
        all_articles.extend(pubmed_fetch(new_pmids))
        time.sleep(0.35)
        if len(all_articles) >= MAX_TOTAL_ARTICLES:
            break
    return all_articles[:MAX_TOTAL_ARTICLES]


# ── Claude ────────────────────────────────────────────────────────────────────

def summarize_with_claude(articles: list[dict]) -> dict:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    today  = datetime.now(BKK_TZ).strftime("%d %B %Y")

    article_block = "\n\n".join([
        f"[{i}] PMID {a['pmid']}\n"
        f"Title: {a['title']}\n"
        f"Authors: {a['authors']}\n"
        f"Journal: {a['journal']} ({a['year']})\n"
        f"Abstract: {a['abstract']}\n"
        f"URL: {a['url']}"
        for i, a in enumerate(articles, 1)
    ])

    system_prompt = """You are an expert oncologist providing a concise daily literature briefing 
to a senior oncologist colleague at Bumrungrad International Hospital. Be clinically precise 
but efficient. Focus on practice-changing findings, novel mechanisms, and key survival/response data."""

    user_prompt = f"""Date: {today}

Analyze these PubMed oncology articles and respond with ONLY a valid JSON object — no markdown, no backticks, no explanation — with this exact structure:
{{
  "top_pick": {{
    "pmid": "...",
    "title": "...",
    "summary": "2-3 sentences: what was done, key result, clinical implication",
    "url": "..."
  }},
  "quick_reads": [
    {{"pmid": "...", "title": "...", "one_liner": "key finding in one sentence", "url": "..."}}
  ],
  "bottom_line": "1-2 sentences on today's overall theme or pattern across the literature",
  "article_count": {len(articles)}
}}

--- ARTICLES ---
{article_block}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Supabase ──────────────────────────────────────────────────────────────────

def log_to_supabase(articles: list[dict], summary: dict) -> None:
    run_id = hashlib.md5(
        datetime.now(BKK_TZ).isoformat().encode()
    ).hexdigest()[:12]

    payload = {
        "run_id":         run_id,
        "run_at":         datetime.now(BKK_TZ).isoformat(),
        "articles_count": len(articles),
        "pmids":          [a["pmid"] for a in articles],
        "summary":        json.dumps(summary),
        "line_delivered": True,
        "queries_used":   SEARCH_QUERIES,
    }

    url = f"{SUPABASE_URL}/rest/v1/oncology_briefings"
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    if r.status_code in (200, 201):
        print(f"  ✓ Saved to Supabase (run_id: {run_id})")
    else:
        print(f"  ✗ Supabase error {r.status_code}: {r.text}")
        raise RuntimeError(f"Supabase write failed: {r.text}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*50}")
    print(f"Oncology Briefing — {datetime.now(BKK_TZ).strftime('%Y-%m-%d %H:%M')} BKK")
    print(f"{'='*50}")

    print("\n[1/3] Fetching PubMed articles…")
    articles = collect_unique_articles()
    if not articles:
        print("  ⚠ No articles found today.")
        log_to_supabase([], {
            "top_pick": None,
            "quick_reads": [],
            "bottom_line": "No new oncology articles matched today's search criteria.",
            "article_count": 0
        })
        return
    print(f"  ✓ {len(articles)} articles collected")

    print("\n[2/3] Summarizing with Claude…")
    summary = summarize_with_claude(articles)
    print(f"  ✓ Summary generated")

    print("\n[3/3] Saving to Supabase…")
    log_to_supabase(articles, summary)

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
