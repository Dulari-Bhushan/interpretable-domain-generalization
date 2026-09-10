"""Plan 09: shared helpers for the per-VLM concept-generation scripts.

Keeps the prompt, image loading, response parsing, and per-class raw-output
storage identical across Claude/Gemini/open-source runs so the only variable
between them is the model itself.
"""
import ast
import base64
import json
import re
from pathlib import Path

CUB_DIR = Path(__file__).parent
IMAGES_DIR = CUB_DIR / "CUB_200_2011" / "images"
MANIFEST_PATH = CUB_DIR / "vlm_concepts" / "sampled_images.json"
RAW_DIR = CUB_DIR / "vlm_concepts" / "raw"

PROMPT_TEMPLATE = """You are looking at {n} photos of a "{class_name}", a bird species.

Based ONLY on what you can actually see in these specific images, list the visual \
features that would help distinguish this species from other bird species — for \
example bill shape/color, plumage color and pattern by body part (crown, nape, \
throat, breast, back, wing, tail, leg), eye color, and overall body size/shape.

Rules:
- Write each feature as a short, atomic phrase (e.g. "a curved bill", "a rufous \
coloured crown") — not a full sentence.
- Only include features you can actually observe in these images. Do not rely on \
general knowledge about this species if it is not visible in the photos shown.
- Return between 8 and 15 phrases.
- Respond with ONLY a JSON array of strings, no other text, no markdown code fences.
"""


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_image_bytes(image_path):
    with open(IMAGES_DIR / image_path, "rb") as f:
        return f.read()


def image_to_base64(image_path):
    return base64.standard_b64encode(load_image_bytes(image_path)).decode("utf-8")


def build_prompt(class_name, n_images):
    return PROMPT_TEMPLATE.format(n=n_images, class_name=class_name)


def parse_concept_list(raw_text):
    """Best-effort parse of a JSON array of strings out of a model response,
    tolerating stray markdown fences, leading/trailing prose, or a response
    that got cut off before its closing bracket (open-source models under a
    token budget, or a degenerate-repetition run, can do this)."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        block = match.group(0)
        try:
            parsed = json.loads(block)
            return _clean_phrases(parsed)
        except json.JSONDecodeError:
            pass
        try:
            # Some models mix single/double quotes or leave trailing commas
            # (invalid JSON but valid Python literal syntax) — Python's own
            # parser tolerates both.
            parsed = ast.literal_eval(block)
            if isinstance(parsed, list):
                return _clean_phrases(parsed)
        except (ValueError, SyntaxError, TypeError):
            # TypeError: e.g. a malformed set literal containing an
            # unhashable dict (ast.literal_eval raises this itself while
            # building the set, not just on unparseable syntax).
            pass

    # Fallback: not valid JSON or Python syntax (some models emit malformed
    # {"feature": "..."} objects with stray/unbalanced quotes). Pull out
    # double-quoted substrings that aren't themselves a JSON *key* (i.e. not
    # immediately followed by a colon) — this recovers "feature": "value"
    # values even when the surrounding object is broken, without hardcoding
    # "feature" as a specific key name some other model might not use.
    # Quotes are paired strictly left-to-right first (unambiguous), then
    # filtered by lookahead — doing the lookahead as part of the pairing
    # regex itself misaligns every subsequent pair once one is rejected,
    # since a failed match backtracks by one character instead of skipping
    # the whole rejected span.
    start = text.rfind("[")
    tail = text[start + 1 :] if start != -1 else text
    quoted = [
        m.group(1)
        for m in re.finditer(r'"([^"]*)"', tail)
        if not re.match(r"\s*:", tail[m.end() : m.end() + 3])
    ]
    deduped = []
    for q in quoted:
        if not deduped or deduped[-1] != q:
            deduped.append(q)
    return _clean_phrases(deduped)


def _looks_like_real_phrase(p):
    # Drops leftover JSON/Python punctuation ("}", ":", ",", "") that the
    # regex/literal-eval fallbacks can pick up from malformed model output.
    return len(re.sub(r"[^A-Za-z]", "", p)) >= 3


def _clean_phrases(items):
    out = []
    for p in items:
        if isinstance(p, dict):
            # Some models emit {"feature": "..."} instead of a plain string
            # despite instructions — take the first string value found.
            str_vals = [v for v in p.values() if isinstance(v, str) and v.strip()]
            p = str_vals[0] if str_vals else ""
        elif isinstance(p, (list, tuple, set, frozenset)):
            # ast.literal_eval turns a stray {"phrase"} into a Python set.
            str_vals = [v for v in p if isinstance(v, str) and v.strip()]
            p = str_vals[0] if str_vals else ""
        p = str(p).strip()
        # Strip a matched pair of wrapping quotes (not a lone leading/trailing
        # one — stripping independently from each end would eat the "'" off
        # an apostrophe-s artifact like "'s black beak" before the next line
        # gets a chance to recognize and remove it as a whole prefix).
        if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'":
            p = p[1:-1].strip()
        p = re.sub(r"^'s\s+", "", p)  # occasional dangling possessive artifact
        if p and _looks_like_real_phrase(p):
            out.append(p)
    return out


def raw_output_path(vlm_name, class_folder):
    out_dir = RAW_DIR / vlm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{class_folder}.json"


def save_class_result(vlm_name, class_folder, class_name, image_paths, raw_text, concepts):
    path = raw_output_path(vlm_name, class_folder)
    with open(path, "w") as f:
        json.dump(
            {
                "class_folder": class_folder,
                "class_name": class_name,
                "image_paths": image_paths,
                "raw_response": raw_text,
                "concepts": concepts,
            },
            f,
            indent=2,
        )


def already_done(vlm_name, class_folder):
    return raw_output_path(vlm_name, class_folder).exists()
