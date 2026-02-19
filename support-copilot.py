import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://api.concentrate.ai"  
RESPONSES_ENDPOINT = f"{BASE_URL}/v1/responses"  


def require_api_key() -> str:
    key = os.getenv("CONCENTRATE_API_KEY")
    if not key:
        raise SystemExit(
            "Missing CONCENTRATE_API_KEY. Set it in your env:\n"
            "  export CONCENTRATE_API_KEY='...'\n"
        )
    return key


def load_long_context(kb_dir: Path) -> str:
    """Loads all KB files into one long context window."""
    parts: List[str] = []
    for p in sorted(kb_dir.glob("**/*")):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            parts.append(f"\n\n### FILE: {p.name}\n{text}")
    if not parts:
        raise SystemExit(f"No .md/.txt files found in {kb_dir}")
    return "\n".join(parts).strip()


def call_concentrate_response(
    api_key: str,
    model: str,
    input_payload: Any,
    *,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_output_tokens: int = 800,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",  # :contentReference[oaicite:6]{index=6}
        "Content-Type": "application/json",
    }

    body = {
        "model": model,  # required 
        "input": input_payload,  # required 
        "temperature": temperature,  
        "top_p": top_p,  
        "max_output_tokens": max_output_tokens,  
        "stream": False,
    }

    t0 = time.time()
    r = requests.post(RESPONSES_ENDPOINT, headers=headers, json=body, timeout=120)
    latency_ms = int((time.time() - t0) * 1000)

    data = None
    try:
        data = r.json()
    except Exception:
        pass

    if r.status_code >= 400:
        raise RuntimeError(
            f"Concentrate API error {r.status_code}: {r.text[:2000]}"
        )

    data["_client_meta"] = {"latency_ms": latency_ms, "model": model}
    return data


def extract_text(response_json: Dict[str, Any]) -> str:
    """
    Concentrate returns a normalized 'output' array containing messages with content blocks.
    We pull any output_text blocks.
    """
    out = []
    for item in response_json.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))
    return "\n".join(out).strip()


def jsonl_log(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_support_answer_messages(customer_email: str, kb_context: str) -> List[Dict[str, Any]]:
    system = (
        "You are a customer support assistant responding to customers emailing the help center for Olivetto olive oil importers.\n"
        "Send a polite response to the customer.\n"
        "If they ask questions, use ONLY the provided Knowledge Base context.\n"
        "If the KB does not contain enough information to answer safely, you may ask 1-2 clarifying questions.\n"
        "Return your response in JSON with keys:\n"
        "  short_answer, needs_clarification (true/false), clarifying_questions (array), kb_quotes (array of exact quotes used)\n"
    )
    user = (
        "CUSTOMER EMAIL:\n"
        f"{customer_email}\n\n"
        "KNOWLEDGE BASE (use as grounding context):\n"
        f"{kb_context}\n"
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system}]},
        {"role": "user", "content": [{"type": "input_text", "text": user}]},
    ]


def build_reviewer_messages(customer_email: str, kb_context: str, draft_json: str) -> List[Dict[str, Any]]:
    system = (
        "You are a strict support QA reviewer analyzing replies sent to customers who have emailed the help center for Olivetto olive oil importers.\n"
        "Check the draft for:\n"
        "1) Unsupported claims (not in KB)\n"
        "2) Missing/weak grounding (should quote KB)\n"
        "3) Tone issues (too cold, too verbose, too apologetic)\n"
        "Return JSON with keys:\n"
        "  unsupported_claims (array), missing_grounding (array), tone_notes (array), recommended_changes (array), verdict (\"ok\"|\"needs_human\"|\"revise\")\n"
    )
    user = (
        "CUSTOMER EMAIL:\n"
        f"{customer_email}\n\n"
        "KNOWLEDGE BASE:\n"
        f"{kb_context}\n\n"
        "DRAFT ANSWER JSON:\n"
        f"{draft_json}\n"
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system}]},
        {"role": "user", "content": [{"type": "input_text", "text": user}]},
    ]


def build_final_email_messages(customer_email: str, kb_context: str, draft_json: str, review_json: str) -> List[Dict[str, Any]]:
    system = (
        "You are a customer support agent writing the final reply to the customer who emailed the help center for Olivetto olive oil importers.\n"
        "Use the KB for all factual/policy statements.\n"
        "Incorporate reviewer feedback.\n"
        "Keep it concise, warm, and actionable.\n"
        "Output ONLY the email body (no JSON)."
    )
    user = (
        "CUSTOMER EMAIL:\n"
        f"{customer_email}\n\n"
        "KNOWLEDGE BASE:\n"
        f"{kb_context}\n\n"
        "DRAFT ANSWER JSON:\n"
        f"{draft_json}\n\n"
        "REVIEW JSON:\n"
        f"{review_json}\n"
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system}]},
        {"role": "user", "content": [{"type": "input_text", "text": user}]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="Path to customer email .txt")
    ap.add_argument("--kb", required=True, help="Path to kb directory")
    ap.add_argument("--openai_model", default="openai/gpt-5.2", help="Provider-prefixed model")  
    ap.add_argument("--anthropic_model", default="anthropic/claude-opus-4.5", help="Provider-prefixed model")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    args = ap.parse_args()

    api_key = require_api_key()
    email_text = Path(args.email).read_text(encoding="utf-8", errors="ignore").strip()
    kb_context = load_long_context(Path(args.kb))

    outdir = Path(args.outdir)
    log_path = outdir / "run_log.jsonl"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Draft answer (OpenAI)
    draft_messages = build_support_answer_messages(email_text, kb_context)
    draft_resp = call_concentrate_response(
        api_key,
        args.openai_model,
        draft_messages,
        temperature=0.8,
        max_output_tokens=900,
        metadata={"stage": "draft_answer"},
    )
    jsonl_log(log_path, {"stage": "draft_answer", "response": draft_resp})
    draft_text = extract_text(draft_resp)
    (outdir / "answer_draft.json").write_text(draft_text + "\n", encoding="utf-8")

    # 2) Review (Anthropic)
    review_messages = build_reviewer_messages(email_text, kb_context, draft_text)
    review_resp = call_concentrate_response(
        api_key,
        args.anthropic_model,
        review_messages,
        temperature=0.2,
        max_output_tokens=700,
        metadata={"stage": "review"},
    )
    jsonl_log(log_path, {"stage": "review", "response": review_resp})
    review_text = extract_text(review_resp)
    (outdir / "review.json").write_text(review_text + "\n", encoding="utf-8")

    # 3) Final email (OpenAI)
    final_messages = build_final_email_messages(email_text, kb_context, draft_text, review_text)
    final_resp = call_concentrate_response(
        api_key,
        args.openai_model,
        final_messages,
        temperature=0.4,
        max_output_tokens=600,
        metadata={"stage": "final_email"},
    )
    jsonl_log(log_path, {"stage": "final_email", "response": final_resp})
    final_email = extract_text(final_resp)
    (outdir / "email_final.txt").write_text(final_email + "\n", encoding="utf-8")

    print("\n=== FINAL EMAIL ===\n")
    print(final_email)
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
