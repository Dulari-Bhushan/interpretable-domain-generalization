"""Per-image LanCE vs VLM agreement pilot - shared prompt/parsing helpers for
the VLM side. Companion to ../agreement_pilot_extract_lance.py (LanCE side).

Unlike Plan 09's concept-bank generation (which asked a VLM to describe a
*class* from several training images, pooled once), this asks a VLM to look
at one specific *test* image and (a) classify it among the same 200 species
LanCE uses, (b) self-report a confidence, and (c) list the concepts that led
to that call - so it can be compared image-by-image against LanCE's own
already-computed prediction/confidence/top-activated-concepts for that same
image.
"""
import ast
import json
import re
from pathlib import Path

CUB_DIR = Path(__file__).parent
IMAGES_DIR = CUB_DIR / "CUB_200_2011" / "images"

PROMPT_TEMPLATE = """You are looking at one photo of a bird. Based ONLY on what you can \
actually see in this image, identify which of the following {n} species it most \
likely is, and list the visual features that led you to that conclusion.

Candidate species (you must pick exactly one, copied EXACTLY as written below):
{class_list}

Respond with ONLY a JSON object, no other text, no markdown code fences, in exactly \
this structure:
{{
  "predicted_class": "<one of the species names above, copied exactly>",
  "confidence": <integer 0-100, your own confidence in this identification>,
  "concepts": ["<short atomic phrase>", "..."]
}}
List 5-10 concise phrases in "concepts" describing what you actually saw in this \
image that led to your identification (e.g. "a curved bill", "a black coloured \
crown") - not general knowledge about the species you named.
"""


def build_prompt(class_names):
    class_list = "\n".join(class_names)
    return PROMPT_TEMPLATE.format(n=len(class_names), class_list=class_list)


CONSTRAINED_PROMPT_TEMPLATE = """You are looking at one photo of a bird. Based ONLY on what you can \
actually see in this image, identify which of the following {n} species it most \
likely is.

Candidate species (you must pick exactly one, copied EXACTLY as written below):
{class_list}

Then, from the fixed list of {m} concept phrases below (this is NOT a list you can add \
to - pick only from what's written here, do not invent your own wording), select the \
5-10 phrases that best describe what you actually see in THIS image supporting your \
identification:
{concept_list}

Respond with ONLY a JSON object, no other text, no markdown code fences, in exactly \
this structure:
{{
  "predicted_class": "<one of the species names above, copied exactly>",
  "confidence": <integer 0-100, your own confidence in this identification>,
  "concepts": ["<a phrase copied EXACTLY from the concept list above>", "..."]
}}
Every string in "concepts" must be copied character-for-character from the concept \
list given above - do not paraphrase, shorten, or write a new phrase.
"""


def build_prompt_constrained(class_names, concept_names):
    class_list = "\n".join(class_names)
    concept_list = "\n".join(concept_names)
    return CONSTRAINED_PROMPT_TEMPLATE.format(
        n=len(class_names), class_list=class_list, m=len(concept_names), concept_list=concept_list
    )


def _normalize_name(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    # crude plural-strip: a model saying "hummingbirds"/"blackbirds" for the
    # singular class name "hummingbird"/"blackbird" is a real match, not a miss.
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.split())


def _extract_fields_by_type(obj, class_names):
    """Pick fields by VALUE TYPE rather than exact key name - a model asked
    for "predicted_class"/"confidence"/"concepts" may instead write
    "predicted_species"/"confidence_level"/"concernpts" (an actual typo seen
    from Qwen2-VL) while still nesting a string, a number, and a list in the
    same three slots. Type-based extraction survives that; exact-key lookup
    doesn't."""
    concepts = None
    confidence = None
    string_values = []
    for v in obj.values():
        if isinstance(v, list) and concepts is None:
            concepts = [str(c).strip() for c in v if str(c).strip()]
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and confidence is None:
            confidence = float(v)
        elif isinstance(v, str) and v.strip():
            string_values.append(v.strip())

    predicted_class = None
    if class_names:
        lookup = {_normalize_name(c): c for c in class_names}
        for v in string_values:
            if _normalize_name(v) in lookup:
                predicted_class = lookup[_normalize_name(v)]
                break
    if predicted_class is None and string_values:
        predicted_class = string_values[0]

    return predicted_class, confidence, (concepts or [])


def parse_classification_response(raw_text, class_names=None):
    """Returns (predicted_class: str|None, confidence: float|None, concepts: list[str]).
    class_names, if given, is used to match a possibly-misnamed-key string
    field against the real candidate list (case/punctuation-insensitive)."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    obj = None
    if match:
        block = match.group(0)
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(block)
            except (ValueError, SyntaxError, TypeError):
                obj = None

    if isinstance(obj, dict):
        return _extract_fields_by_type(obj, class_names)

    # Fallback: pull fields out by regex if the object itself didn't parse.
    pred_match = re.search(r'"predicted_class"\s*:\s*"([^"]*)"', text)
    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    concepts = re.findall(r'"([^"]{3,})"', text)
    predicted_class = pred_match.group(1).strip() if pred_match else None
    confidence = float(conf_match.group(1)) if conf_match else None
    # crude fallback concepts: drop the predicted_class string itself if it leaked in
    concepts = [c for c in concepts if c != predicted_class][:10]
    return predicted_class, confidence, concepts


def parse_constrained_response(raw_text, class_names, concept_names):
    """Same as parse_classification_response, but "concepts" is validated
    against concept_names (normalized match) instead of accepted as free
    text - the model was told to select from a fixed vocabulary, not write
    its own, so a selection that doesn't match anything in that vocabulary
    is a real instruction-following failure, not a valid new concept.

    Returns (predicted_class, confidence, valid_concepts, n_invalid_selections).
    """
    predicted_class, confidence, raw_concepts = parse_classification_response(raw_text, class_names)

    concept_lookup = {_normalize_name(c): c for c in concept_names}
    valid_concepts = []
    n_invalid = 0
    for phrase in raw_concepts:
        key = _normalize_name(phrase)
        if key in concept_lookup:
            canonical = concept_lookup[key]
            if canonical not in valid_concepts:
                valid_concepts.append(canonical)
        else:
            n_invalid += 1

    return predicted_class, confidence, valid_concepts, n_invalid


def load_lance_pilot(path=None):
    path = path or (Path(__file__).parent.parent.parent / "agreement_pilot_lance.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
