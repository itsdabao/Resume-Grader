"""
CV Scorer — LLM-powered scoring pipeline with phrase-level annotations.

Uses Groq as primary LLM, Google Gemini as fallback.
Employs multi-criteria rubric scoring for reliability.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema (matches frontend types exactly)
# ---------------------------------------------------------------------------

class AnnotationOut(BaseModel):
    """A phrase-level annotation on the CV text."""
    text: str = Field(description="The exact phrase from the CV being annotated")
    kind: str = Field(description="'bonus' for positive signal, 'penalty' for negative signal")
    tip: str = Field(description="Short tooltip explaining why this is a bonus or penalty")


class CVReviewResult(BaseModel):
    """Full review result for a single CV."""
    score: int = Field(ge=0, le=100, description="Overall fit score 0-100")
    strengths: List[str] = Field(default_factory=list, description="Key strengths (3-5 bullet points)")
    weaknesses: List[str] = Field(default_factory=list, description="Key weaknesses (2-4 bullet points)")
    reasoning: str = Field(default="", description="Detailed reasoning paragraph")
    annotations: List[AnnotationOut] = Field(default_factory=list, description="Phrase-level annotations")


class CandidateResult(BaseModel):
    """Lightweight candidate info returned from batch upload."""
    id: str
    name: str
    role: str = "Candidate"
    score: int = Field(ge=0, le=100)
    cv_text: str = ""


# ---------------------------------------------------------------------------
# In-memory result cache (no persistence)
# ---------------------------------------------------------------------------

_CACHE: Dict[str, CVReviewResult] = {}
_CACHE_MAX = 200


def _cache_key(cv_text: str, jd_text: str) -> str:
    h = hashlib.sha256((cv_text[:2000] + "||" + jd_text[:2000]).encode()).hexdigest()[:24]
    return h


def _cache_get(key: str) -> Optional[CVReviewResult]:
    return _CACHE.get(key)


def _cache_put(key: str, result: CVReviewResult) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        # Evict oldest 25%
        keys = list(_CACHE.keys())
        for k in keys[: len(keys) // 4]:
            _CACHE.pop(k, None)
    _CACHE[key] = result


# ---------------------------------------------------------------------------
# LLM client helpers (independent from Settings.llm)
# ---------------------------------------------------------------------------

def _call_groq(prompt: str, *, max_tokens: int = 2048, temperature: float = 0.15) -> str:
    """
    Call Groq via OpenAI-compatible API directly (no LlamaIndex dependency).
    Returns raw text response.
    """
    import httpx

    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    base_url = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").strip().rstrip("/")
    model = (os.getenv("GROQ_MODEL") or "meta-llama/llama-4-scout-17b-16e-instruct").strip()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=90.0) as client:
        resp = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )

    if resp.status_code >= 400:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()


def _call_gemini(prompt: str, *, max_tokens: int = 2048, temperature: float = 0.15) -> str:
    """
    Call Google Gemini via REST API directly.
    Returns raw text response.
    """
    import httpx

    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    model = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }

    with httpx.Client(timeout=90.0) as client:
        resp = client.post(url, json=payload)

    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response structure: {json.dumps(data)[:500]}")


def _call_llm(prompt: str, **kwargs) -> str:
    """Call LLM with Groq primary, Gemini fallback."""
    errors = []

    # Try Groq first
    if os.getenv("GROQ_API_KEY"):
        try:
            return _call_groq(prompt, **kwargs)
        except Exception as e:
            errors.append(f"Groq: {e}")
            logger.warning("Groq call failed: %s — trying Gemini fallback", e)

    # Try Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            return _call_gemini(prompt, **kwargs)
        except Exception as e:
            errors.append(f"Gemini: {e}")
            logger.warning("Gemini call failed: %s", e)

    raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Scoring prompt (multi-criteria rubric)
# ---------------------------------------------------------------------------

_SCORING_PROMPT = """\
You are an expert AI CV/Resume reviewer. Analyze the candidate's CV against the job description (if provided) and produce a structured evaluation.

## Scoring Rubric (Multi-Criteria, each 0-25 points, total 0-100):
1. **Skill Match** (0-25): How well do the candidate's technical skills and tools match the requirements?
2. **Experience Relevance** (0-25): How relevant is their work experience (years, industry, role level)?
3. **Achievement Quality** (0-25): Are accomplishments quantified with metrics? Do they demonstrate real impact?
4. **Presentation** (0-25): Is the CV well-structured, concise, free of vague language and red flags?

## Output Requirements:
Return a single JSON object with these exact fields:
- "skill_match": integer 0-25
- "experience_relevance": integer 0-25
- "achievement_quality": integer 0-25
- "presentation": integer 0-25
- "score": integer 0-100 (sum of the 4 criteria above)
- "strengths": array of 3-5 short bullet-point strings
- "weaknesses": array of 2-4 short bullet-point strings
- "reasoning": string — 2-3 sentences explaining overall fit
- "annotations": array of 5-10 objects, each with:
  - "text": the EXACT phrase copied from the CV (do NOT paraphrase)
  - "kind": "bonus" or "penalty"
  - "tip": 1-sentence tooltip explaining why this phrase is a bonus/penalty

## Annotation Guidelines:
- Pick 5-10 of the most impactful phrases
- "bonus" annotations: quantified achievements, relevant skills, leadership signals, strong keywords
- "penalty" annotations: vague/generic claims, buzzwords without evidence, gaps, red flags, outdated tech
- The "text" field MUST be an exact substring of the CV text (case-sensitive match)

{jd_section}

## CANDIDATE CV:
```
{cv_text}
```

Return ONLY the JSON object. No markdown fences, no explanation outside the JSON.
"""

_JD_SECTION = """\
## JOB DESCRIPTION:
```
{jd_text}
```
"""

_NO_JD_SECTION = """\
## JOB DESCRIPTION:
No specific job description provided. Evaluate the CV on general professional quality and clarity.
"""


def _build_scoring_prompt(cv_text: str, jd_text: str) -> str:
    """Build the full scoring prompt."""
    cv_trimmed = cv_text[:6000]  # Keep under context limits
    if jd_text.strip():
        jd_section = _JD_SECTION.format(jd_text=jd_text[:2000])
    else:
        jd_section = _NO_JD_SECTION

    return _SCORING_PROMPT.format(cv_text=cv_trimmed, jd_section=jd_section)


# ---------------------------------------------------------------------------
# Quick scoring (for batch upload — faster, less detail)
# ---------------------------------------------------------------------------

_QUICK_SCORING_PROMPT = """\
You are an expert CV reviewer. For EACH resume below, provide a quick assessment.

Return a JSON object with field "candidates" containing an array.
Each element has:
- "index": integer (0-based, matching the resume order below)
- "name": string (the candidate's full name extracted from the CV)
- "role": string (their most recent or primary job title)
- "score": integer 0-100 (overall quality score)

{jd_section}

## RESUMES:
{resumes_block}

Return ONLY the JSON object. No markdown fences.
"""


def _build_quick_prompt(cv_texts: List[Tuple[int, str]], jd_text: str) -> str:
    """Build prompt for batch quick scoring."""
    resumes = []
    for idx, text in cv_texts:
        trimmed = text[:2000]
        resumes.append(f"--- Resume #{idx} ---\n{trimmed}\n")

    resumes_block = "\n".join(resumes)
    if jd_text.strip():
        jd_section = _JD_SECTION.format(jd_text=jd_text[:1500])
    else:
        jd_section = _NO_JD_SECTION

    return _QUICK_SCORING_PROMPT.format(resumes_block=resumes_block, jd_section=jd_section)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """
    Robustly extract JSON from LLM response.
    Handles markdown fences, leading/trailing text, etc.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    # Find the outermost { ... }
    start = text.find("{")
    if start >= 0:
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_cv(cv_text: str, jd_text: str = "") -> CVReviewResult:
    """
    Score a single CV with full detail (strengths, weaknesses, reasoning, annotations).
    Uses multi-criteria rubric via LLM.
    Results are cached in-memory.
    """
    if not cv_text.strip():
        return CVReviewResult(
            score=0,
            strengths=[],
            weaknesses=["CV text is empty or could not be extracted"],
            reasoning="Unable to evaluate: no text content found in the uploaded file.",
            annotations=[],
        )

    # Check cache
    ck = _cache_key(cv_text, jd_text)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    prompt = _build_scoring_prompt(cv_text, jd_text)

    try:
        raw = _call_llm(prompt, max_tokens=2048, temperature=0.15)
        data = _extract_json(raw)
    except Exception as e:
        logger.error("LLM scoring failed: %s", e)
        return _fallback_score(cv_text)

    try:
        # Validate and clamp score
        sub_scores = {
            "skill_match": _clamp(data.get("skill_match", 0), 0, 25),
            "experience_relevance": _clamp(data.get("experience_relevance", 0), 0, 25),
            "achievement_quality": _clamp(data.get("achievement_quality", 0), 0, 25),
            "presentation": _clamp(data.get("presentation", 0), 0, 25),
        }
        total = sum(sub_scores.values())
        # Use LLM's total if close to sum, otherwise use computed sum
        llm_total = _clamp(data.get("score", total), 0, 100)
        if abs(llm_total - total) <= 5:
            total = llm_total

        strengths = _ensure_str_list(data.get("strengths", []))[:5]
        weaknesses = _ensure_str_list(data.get("weaknesses", []))[:4]
        reasoning = str(data.get("reasoning", "") or "")

        # Validate annotations — only keep those whose text actually appears in the CV
        raw_annotations = data.get("annotations", [])
        annotations = []
        for ann in raw_annotations:
            if not isinstance(ann, dict):
                continue
            ann_text = str(ann.get("text", "")).strip()
            ann_kind = str(ann.get("kind", "")).strip().lower()
            ann_tip = str(ann.get("tip", "")).strip()
            if not ann_text or ann_kind not in ("bonus", "penalty") or not ann_tip:
                continue
            # Verify the annotation text actually exists in the CV
            if ann_text in cv_text:
                annotations.append(AnnotationOut(text=ann_text, kind=ann_kind, tip=ann_tip))
            else:
                # Try case-insensitive match
                if ann_text.lower() in cv_text.lower():
                    # Find the actual case from CV
                    idx = cv_text.lower().find(ann_text.lower())
                    actual = cv_text[idx : idx + len(ann_text)]
                    annotations.append(AnnotationOut(text=actual, kind=ann_kind, tip=ann_tip))

        result = CVReviewResult(
            score=total,
            strengths=strengths,
            weaknesses=weaknesses,
            reasoning=reasoning,
            annotations=annotations[:10],  # Cap at 10
        )
        _cache_put(ck, result)
        return result

    except Exception as e:
        logger.error("Failed to parse LLM scoring result: %s", e)
        return _fallback_score(cv_text)


def quick_score_batch(
    cv_texts: List[Tuple[str, str]],  # [(filename, text), ...]
    jd_text: str = "",
) -> List[CandidateResult]:
    """
    Quick-score multiple CVs in one LLM call (batch).
    Returns sorted list of CandidateResult (highest score first).
    """
    if not cv_texts:
        return []

    indexed = [(i, text) for i, (_, text) in enumerate(cv_texts)]
    filenames = [fn for fn, _ in cv_texts]

    # If too many, split into batches of 5
    all_results: List[CandidateResult] = []
    batch_size = 5
    for batch_start in range(0, len(indexed), batch_size):
        batch = indexed[batch_start : batch_start + batch_size]
        batch_fns = filenames[batch_start : batch_start + batch_size]
        results = _quick_score_one_batch(batch, batch_fns, jd_text)
        all_results.extend(results)

    # Sort by score descending
    all_results.sort(key=lambda c: c.score, reverse=True)
    return all_results


def _quick_score_one_batch(
    indexed_texts: List[Tuple[int, str]],
    filenames: List[str],
    jd_text: str,
) -> List[CandidateResult]:
    """Score a single batch of CVs."""
    from app.services.cv_parser import guess_candidate_name

    prompt = _build_quick_prompt(indexed_texts, jd_text)

    try:
        raw = _call_llm(prompt, max_tokens=1024, temperature=0.1)
        data = _extract_json(raw)
    except Exception as e:
        logger.error("Quick scoring LLM call failed: %s — using heuristic fallback", e)
        return _quick_fallback(indexed_texts, filenames)

    candidates_data = data.get("candidates", [])
    if not isinstance(candidates_data, list):
        candidates_data = []

    results: List[CandidateResult] = []
    seen_indices = set()

    for item in candidates_data:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(indexed_texts):
            continue
        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        global_idx = indexed_texts[idx][0]
        name = str(item.get("name", "")).strip()
        role = str(item.get("role", "")).strip() or "Candidate"
        score = _clamp(item.get("score", 50), 0, 100)

        if not name:
            name = guess_candidate_name(indexed_texts[idx][1], filenames[idx])

        results.append(CandidateResult(
            id=f"cv-{global_idx}-{int(time.time())}",
            name=name,
            role=role,
            score=score,
            cv_text=indexed_texts[idx][1],
        ))

    # Fill in any missing indices
    for i, (global_idx, text) in enumerate(indexed_texts):
        if i not in seen_indices:
            name = guess_candidate_name(text, filenames[i])
            results.append(CandidateResult(
                id=f"cv-{global_idx}-{int(time.time())}",
                name=name,
                role="Candidate",
                score=50,
                cv_text=text,
            ))

    return results


# ---------------------------------------------------------------------------
# Fallback scorers (when LLM fails)
# ---------------------------------------------------------------------------

def _fallback_score(cv_text: str) -> CVReviewResult:
    """
    Heuristic fallback when LLM is unavailable.
    Scores based on text length, keyword presence, and structure.
    """
    text = cv_text.lower()
    score = 40  # baseline

    # Length bonus (well-detailed CVs)
    if len(cv_text) > 1500:
        score += 10
    if len(cv_text) > 3000:
        score += 5

    # Quantified achievements
    numbers = len(re.findall(r"\d+%|\d+x|\$\d+|[0-9]+\+", text))
    score += min(numbers * 3, 15)

    # Keywords
    positive_kws = ["led", "built", "shipped", "reduced", "improved", "managed", "designed",
                    "architected", "mentored", "scaled", "optimized", "implemented"]
    kw_hits = sum(1 for kw in positive_kws if kw in text)
    score += min(kw_hits * 2, 10)

    # Penalty for vague language
    vague_kws = ["various", "helped with", "responsible for", "assisted", "participated"]
    vague_hits = sum(1 for kw in vague_kws if kw in text)
    score -= min(vague_hits * 3, 10)

    score = _clamp(score, 10, 95)

    return CVReviewResult(
        score=score,
        strengths=["CV text was successfully extracted"],
        weaknesses=["LLM scoring unavailable — using heuristic fallback"],
        reasoning=f"Heuristic score based on text analysis: length, quantified achievements, and keyword presence. Score: {score}/100. For accurate scoring, ensure Groq or Gemini API keys are configured.",
        annotations=[],
    )


def _quick_fallback(
    indexed_texts: List[Tuple[int, str]],
    filenames: List[str],
) -> List[CandidateResult]:
    """Heuristic quick scoring when LLM is unavailable."""
    from app.services.cv_parser import guess_candidate_name

    results = []
    for i, (global_idx, text) in enumerate(indexed_texts):
        name = guess_candidate_name(text, filenames[i])
        result = _fallback_score(text)
        results.append(CandidateResult(
            id=f"cv-{global_idx}-{int(time.time())}",
            name=name,
            role="Candidate",
            score=result.score,
            cv_text=text,
        ))
    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(value: Any, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = lo
    return max(lo, min(hi, v))


def _ensure_str_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val if x]
    return []
