import os
import json
import re
from typing import Optional

from google import genai
from google.genai import errors


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")
    return genai.Client(api_key=api_key)


def _try_generate(client: genai.Client, model: str, prompt: str) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return getattr(resp, "text", None) or str(resp)


def _extract_json(text: str) -> dict:
    """
    Gemini kabhi-kabhi JSON ke bahar extra text de deta hai.
    Ye function output se FIRST JSON object extract karke parse karta hai.
    """
    text = (text or "").strip()

    # 1) direct parse try
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) JSON object extract (first {...} block)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model response")

    json_str = match.group(0)

    # 3) parse extracted JSON
    return json.loads(json_str)


def review_code(code: str, pro_prompt: Optional[str] = "") -> dict:
    code = (code or "").strip()
    pro_prompt = (pro_prompt or "").strip()

    if not code:
        return {"ok": False, "error": "No code provided."}

    client = _get_client()

    preferred = os.getenv("GEMINI_MODEL", "").strip()
    model_candidates = [m for m in [preferred] if m] + [
        "gemini-2.5-flash",
        # (optional fallback add kar sakte ho)
        # "gemini-2.0-flash",
        # "gemini-1.5-flash",
    ]

    instruction = f"""
You are CodeRefine AI. You MUST respond ONLY in ENGLISH.
Return output ONLY as valid JSON (no markdown, no triple backticks).

JSON schema:
{{
  "quality": "string (short, simple)",
  "issues": ["string", "..."],
  "suggestions": ["string", "..."],
  "corrected_code": "string (best possible corrected/improved version)",
  "verdict": "string (1 line summary)"
}}

Guidelines:
- Keep it EASY to understand (short sentences).
- If user asked for changes, apply them in corrected_code.
- If language is unclear, assume Python unless code indicates otherwise.
- If code seems like another language, mention it briefly in issues and provide corrected_code in the requested language if user asked.

User requested changes (Pro prompt): {pro_prompt if pro_prompt else "None"}

CODE:
{code}
""".strip()

    last_err = None

    for model in model_candidates:
        try:
            text = _try_generate(client, model, instruction)

            # ✅ FIX: safe JSON parse (extract JSON even if extra text exists)
            data = _extract_json(text)

            # minimal validation
            for k in ["quality", "issues", "suggestions", "corrected_code", "verdict"]:
                if k not in data:
                    raise ValueError(f"Missing key in model response: {k}")

            return {"ok": True, "model": model, "review": data}

        except errors.ClientError as e:
            last_err = str(e)
            continue

        except Exception as e:
            last_err = str(e)
            continue

    return {
        "ok": False,
        "error": f"Unable to generate review. Last error: {last_err}",
        "hint": "Try setting GEMINI_MODEL env var to a model available on your account.",
    }

# uvicorn main:app --reload