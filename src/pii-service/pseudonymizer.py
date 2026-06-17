"""
pseudonymizer.py — token masking logic.

Replaces detected PII with consistent placeholder tokens. The same original value
always maps to the same token, and common variants (names with inter-character
spacing, national IDs with/without separators, etc.) are replaced too.
"""

import logging
import re
import unicodedata
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Detection label -> token prefix. Token format is ``[PREFIX_N]``.
LABEL_TO_TAG_PREFIX = {
    "p_nm": "PERSON",
    "p_rrn": "RRN",       # Korean resident registration number
    "p_ssn": "SSN",       # US social security number
    "p_ph": "PHONE",
    "p_em": "EMAIL",
    "p_add": "ADDRESS",
    "p_ip": "IP",
    "p_acn": "ACCOUNT",
    "p_pp": "PASSPORT",
    "p_dt": "DATE",
    "p_org": "ORG",
    "p_loc": "LOCATION",
    "p_url": "URL",
    "p_card": "CARD",
    "p_dob": "DOB",
    "p_rel": "REL",
}

# Detection label -> human-readable description.
LABEL_DESCRIPTIONS = {
    "p_nm": "name",
    "p_rrn": "resident registration number",
    "p_ssn": "social security number",
    "p_ph": "phone number",
    "p_em": "email",
    "p_add": "address",
    "p_ip": "IP address",
    "p_acn": "account number",
    "p_pp": "passport number",
    "p_dt": "date/time",
    "p_org": "organization",
    "p_loc": "location",
    "p_url": "URL",
    "p_card": "card number",
    "p_dob": "date of birth",
    "p_rel": "family relationship",
}


def build_pseudonym_mapping(entities):
    """Build a token mapping from a list of detected entities.

    The same original text occurring multiple times receives the same token.

    Returns:
        OrderedDict: {original_text: {"tag", "label", "description"}}
    """
    mapping = OrderedDict()
    label_counters = {}

    for entity in entities:
        label = entity["entity_group"]
        word = entity["word"].strip()
        if not word or word in mapping:
            continue

        tag_prefix = LABEL_TO_TAG_PREFIX.get(label, label.upper())
        description = LABEL_DESCRIPTIONS.get(label, label)
        label_counters[label] = label_counters.get(label, 0) + 1
        tag = f"[{tag_prefix}_{label_counters[label]}]"
        mapping[word] = {"tag": tag, "label": label, "description": description}

    return mapping


def pseudonymize_text(text, entities):
    """Replace detected PII in ``text`` with placeholder tokens.

    After replacing each detected entity, the same original value occurring
    elsewhere in the document is also replaced (global string substitution), along
    with common variant spellings.

    Returns:
        tuple: (pseudonymized_text, mapping)
    """
    mapping = build_pseudonym_mapping(entities)
    if not mapping:
        return text, mapping

    # NFC-normalize to avoid missed replacements due to unicode form differences.
    result = unicodedata.normalize("NFC", text)
    # Replace longer values first so a short value is not substituted inside a longer one.
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)

    for original, info in sorted_mapping:
        tag = info["tag"]
        label = info["label"]
        original = unicodedata.normalize("NFC", original)

        if result.count(original) > 0:
            result = result.replace(original, tag)

        for variant in _generate_variants(original, label):
            if result.count(variant) > 0:
                result = result.replace(variant, tag)

    return result, mapping


def _generate_variants(original, label):
    """Generate variant spellings of an entity (spacing / separator differences)."""
    variants = []

    if label == "p_nm":
        # Names with spacing between characters, e.g. OCR'd signatures.
        chars = list(original.replace(" ", ""))
        if len(chars) >= 2:
            for sep in (" ", "  ", "   ", "    ", "\t", "\n"):
                variants.append(sep.join(chars))

    elif label in ("p_rrn", "p_ssn"):
        # National ID with/without spaces around the hyphens.
        normalized = re.sub(r"\s+", "", original)
        if normalized != original:
            variants.append(normalized)
        rrn_match = re.match(r"(\d{6})(-)(\d{7})", normalized)
        if rrn_match:
            g1, g3 = rrn_match.group(1), rrn_match.group(3)
            variants += [f"{g1} - {g3}", f"{g1}-{g3}", f"{g1} -{g3}", f"{g1}- {g3}"]
        ssn_match = re.match(r"(\d{3})-(\d{2})-(\d{4})", normalized)
        if ssn_match:
            a, b, c = ssn_match.groups()
            variants += [f"{a} - {b} - {c}", f"{a}-{b}-{c}"]

    elif label == "p_ph":
        # Phone number with different separators.
        normalized = re.sub(r"[\s\-.]", "", original)
        if len(normalized) == 11:
            variants += [
                f"{normalized[:3]}-{normalized[3:7]}-{normalized[7:]}",
                f"{normalized[:3]} {normalized[3:7]} {normalized[7:]}",
                f"{normalized[:3]}.{normalized[3:7]}.{normalized[7:]}",
                normalized,
            ]

    variants = [v for v in variants if v != original]
    return list(dict.fromkeys(variants))


def reassemble(analysis_result, pii_mappings, national_id_prefixes=("RRN", "SSN")):
    """Restore PII tokens in text via deterministic string replacement.

    National IDs (token prefixes in ``national_id_prefixes``) are restored only
    partially — the leading group is kept and the rest masked with ``*`` — so the
    final document never re-exposes the full identifier.

    Args:
        analysis_result: text containing [PREFIX_N] tokens
        pii_mappings:     list of {"token", "original"}
        national_id_prefixes: token prefixes to partially mask

    Returns:
        tuple: (final_text, replaced_count, unreplaced_tokens)
    """
    prefixes = tuple(national_id_prefixes)
    result_text = analysis_result
    replaced_count = 0

    for mapping in pii_mappings:
        token = f"[{mapping['token']}]"
        original = mapping["original"]

        if mapping["token"].startswith(prefixes) and "-" in original:
            parts = original.split("-")
            masked_tail = "-".join("*" * len(p) for p in parts[1:])
            original = f"{parts[0]}-{masked_tail}"

        count = result_text.count(token)
        if count > 0:
            result_text = result_text.replace(token, original)
            replaced_count += count

    unreplaced = re.findall(r"\[\w+_\d+\]", result_text)
    return result_text, replaced_count, unreplaced
