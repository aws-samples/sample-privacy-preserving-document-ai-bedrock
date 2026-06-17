"""
Language-keyed prompts and regex supplements for the privacy-preserving pipeline.

Everything that is language-specific lives here so the rest of the pipeline stays
language-agnostic. Select a language with the ``lang`` parameter (``"ko"`` or ``"en"``)
that is threaded from ``run_pipeline`` down into detection and analysis.

To add a third language:
  1. Add an entry to ``PII_PROMPTS`` (a detection prompt that emits TSV ``TYPE<TAB>ORIGINAL``).
  2. Add an entry to ``REGEX_SETS`` (a list of ``(pii_type, compiled_pattern)``).
  3. Add an entry to ``ANALYSIS_PROMPTS`` and ``ANALYSIS_WRAPPER``.
  4. Drop a matching ``sample-documents/<lang>/<scenario>.txt`` set.
"""

import re

SUPPORTED_LANGS = ("ko", "en")
DEFAULT_LANG = "ko"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — PII detection prompts (sent to the Qwen sLLM, one chunk at a time).
#
# The prompt instructs the model to scan the document end to end and emit one
# PII item per line as ``TYPE<TAB>ORIGINAL``. TSV (rather than JSON) keeps the
# decode token count low, which shortens wall-clock latency.
#
# NOTE: the ``$TEXT$`` placeholder is used instead of ``{text}`` so the prompt can
# contain literal JSON/braces without colliding with ``str.format``.
# ─────────────────────────────────────────────────────────────────────────────

_PII_PROMPT_KO = """금융 문서에서 개인정보(PII)를 빠짐없이 추출하세요.

다음 순서대로 텍스트를 처음부터 끝까지 스캔하세요:

1단계) RRN(주민등록번호): 6자리-7자리 숫자 패턴(예: 900720-2345678). 마스킹(******)된 것만 제외.
2단계) PHONE(전화번호): 아래 모든 유형을 빠짐없이 추출:
  - 휴대폰: 010-xxxx-xxxx
  - 사무실/지역번호: 02-xxxx-xxxx, 031-xxx-xxxx, 032-xxx-xxxx 등
  - 대표전화: 1588-xxxx, 1577-xxxx, 1599-xxxx 등 (15xx/16xx로 시작하는 4+4자리)
  - 민원전화: 1332 (금융감독원)
3단계) PERSON(이름): 한글 2~4자 인명 + 영문 이름(CHOI YUJIN, Park Jihun 등). 띄어쓴 서명("이 서 연")도 포함. 성명(한글)과 성명(영문)이 있으면 둘 다 추출.
4단계) EMAIL: @가 포함된 이메일 주소
5단계) ADDRESS(주소): 텍스트에 나오는 모든 주소를 추출. 절대 빠뜨리지 마세요:
  - 개인 주소: 아파트, 오피스텔 등 (예: "서울특별시 강남구 테헤란로 123, 현대아파트 201동 504호")
  - 회사/기관 주소: 보험사, 은행, 증권사 본점/지점
  - 문서 하단(footer) 주소: 발행기관 소재지
  - 다음 두 형태를 모두 잡으세요:
    a) 도로명 주소 — "~로/길/대로" + 번지 + 건물명
    b) 지번 주소(legacy) — "~동/리" + 번지(보통 123-45 형태) + 건물명/동/호
  - 동/호 표기는 모두 같은 ADDRESS의 일부로 묶어서 추출
6단계) ACCOUNT(계좌번호): 숫자-숫자-숫자 (10자리+). 사업자등록번호(3-2-5자리, 예: 124-81-00998)는 제외.
7단계) CARD(카드번호): 16자리 카드번호 (4-4-4-4 또는 4444444444444444 형태). 부분 마스킹된 카드번호("1234-****-****-5678")도 모두 추출.
8단계) DOB(생년월일): 생년월일 표기 (YYYY-MM-DD, YYYY.MM.DD, YYYY년 MM월 DD일). RRN과 별도 항목으로 추출.
9단계) REL(가족관계): 가족관계를 나타내는 표현 — "배우자", "남편", "아내", "자녀", "장남", "차녀", "부", "모", "보호자" 등. 관계 표현이 인물 이름과 함께 등장하면 관계어만 추출(인명은 PERSON으로 분리).

진단명·병명·수술명·투약·검사 결과는 의사결정의 근거 데이터로 사용되므로 마스킹 대상에서 제외합니다. 식별자(이름/RRN/DOB/연락처/주소 등)만 분리하여 큰 모델이 "누구의 의료정보인지" 모르게 하는 것이 목적입니다.

[예시 입력]
성명(한글): 홍길동 | 주민등록번호: 850315-1234567 | 생년월일: 1985.03.15
주소: 서울특별시 강남구 역삼로 180, 한국빌딩 12F
전화: 010-1234-5678 | 이메일: hong@test.com
계좌: 110-456-789012 | 보호자: 배우자 김영희(010-9999-0000)

[예시 출력 — TSV 형식, 한 줄에 한 항목, type<TAB>original]
RRN	850315-1234567
DOB	1985.03.15
PERSON	홍길동
PERSON	김영희
REL	배우자
ADDRESS	서울특별시 강남구 역삼로 180, 한국빌딩 12F
PHONE	010-1234-5678
EMAIL	hong@test.com
ACCOUNT	110-456-789012
PHONE	010-9999-0000

중복 제거. 위 예시와 동일한 TSV 형식으로만 출력하세요 (TYPE<TAB>ORIGINAL 한 줄에 하나, 머리말/설명/JSON 금지).

[실제 텍스트]
$TEXT$"""


_PII_PROMPT_EN = """Extract every piece of personally identifiable information (PII) from the financial document below.

Scan the text from beginning to end in the following order:

Step 1) SSN (Social Security Number): the 3-2-4 digit pattern (e.g. 521-84-6390). Skip only fully masked ones (xxx-xx-xxxx).
Step 2) PHONE: extract every form, including:
  - Mobile/landline: (415) 555-2389, 415-555-2389, 415.555.2389
  - Toll-free: 1-800-xxx-xxxx, 1-888-xxx-xxxx
Step 3) PERSON (name): personal names (first + last, e.g. Michael Carter). Include names appearing on signature lines.
Step 4) EMAIL: any email address containing '@'.
Step 5) ADDRESS: extract every address present. Do not miss any:
  - Personal addresses: street, unit/apt, city, state, ZIP (e.g. "1420 Sutter Street, Apt 302, San Francisco, CA 94109")
  - Company/institution addresses (bank, insurer, brokerage branches)
  - Footer addresses of the issuing institution
  - Capture the whole address (street + unit + city + state + ZIP) as one ADDRESS item.
Step 6) ACCOUNT (bank account number): digit-digit-digit groups (10+ digits). Exclude Employer Tax IDs / EINs (2-7 digit, e.g. 36-4512345).
Step 7) CARD (card number): 16-digit card numbers (4-4-4-4 or 4444444444444444). Also extract partially masked cards ("4532-****-****-3471").
Step 8) DOB (date of birth): date-of-birth values (YYYY-MM-DD, MM/DD/YYYY, Month DD, YYYY). Extract as a separate item from SSN.
Step 9) REL (family relationship): relationship terms — "spouse", "husband", "wife", "son", "daughter", "father", "mother", "guardian", etc. When a relationship term appears next to a name, extract only the relationship term (the name goes to PERSON).

Diagnoses, conditions, procedures, medications, and test results are decision-supporting data and are NOT masking targets. The goal is to separate only the identifiers (name/SSN/DOB/contact/address) so the large model cannot tell "whose" medical information it is.

[Example input]
Name: John Doe | SSN: 521-84-6390 | DOB: 1985-03-15
Address: 1420 Sutter Street, Apt 302, San Francisco, CA 94109
Phone: (415) 555-1234 | Email: john@test.com
Account: 110-456-789012 | Guardian: spouse Jane Doe (415) 555-9999

[Example output — TSV format, one item per line, type<TAB>original]
SSN	521-84-6390
DOB	1985-03-15
PERSON	John Doe
PERSON	Jane Doe
REL	spouse
ADDRESS	1420 Sutter Street, Apt 302, San Francisco, CA 94109
PHONE	(415) 555-1234
EMAIL	john@test.com
ACCOUNT	110-456-789012
PHONE	(415) 555-9999

Remove duplicates. Output ONLY in the TSV format above (TYPE<TAB>ORIGINAL, one per line; no preamble, explanation, or JSON).

[Actual text]
$TEXT$"""


PII_PROMPTS = {"ko": _PII_PROMPT_KO, "en": _PII_PROMPT_EN}


# Valid PII types the model is allowed to emit (1:1 with the prompt steps).
# Items whose first TSV column is not in this set are ignored. Keep in sync with
# the pii-service ``QWEN_TYPE_TO_NER_LABEL`` map and ``pseudonymizer`` prefixes.
VALID_PII_TYPES = frozenset({
    "RRN", "SSN", "DOB", "PERSON", "REL", "ADDRESS",
    "PHONE", "EMAIL", "ACCOUNT", "CARD",
})


# ─────────────────────────────────────────────────────────────────────────────
# Regex supplements — catch fixed-format identifiers the model may miss.
# Each entry is ``(pii_type, compiled_pattern)``. The first capture group (or the
# whole match) is used. Per-type post-processing lives in pipeline._supplement_with_regex.
# ─────────────────────────────────────────────────────────────────────────────

_REGEX_KO = [
    ("RRN", re.compile(r'(\d{6})\s*-\s*(\d[\d*]{6})')),
    ("PHONE", re.compile(r'(?<!\d)(0(?:10|[2-6]\d?|70)\s*-?\s*\d{3,4}\s*-\s*\d{4})(?!\d)')),
    ("PHONE", re.compile(r'(?<!\d)((?:15[0-9]{2}|16[0-9]{2}|17[0-9]{2}|18[0-9]{2})\s*-?\s*\d{4})(?!\d)')),
    ("PHONE", re.compile(r'(?<!\d)(1332)(?!\d)')),
    ("PHONE", re.compile(r'(?<!\d)(02\s*-\s*\d{3,4}\s*-\s*\d{4})(?!\d)')),
    ("PHONE", re.compile(r'(?<!\d)(03[1-9]\s*-\s*\d{3,4}\s*-\s*\d{4})(?!\d)')),
    ("EMAIL", re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')),
    ("ACCOUNT", re.compile(r'(?<!\d)(\d{2,4}\s*-\s*\d{2,6}\s*-\s*\d{4,8})(?!\d)')),
    ("ADDRESS", re.compile(
        r'((?:서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시'
        r'|세종특별자치시|경기도|강원특별자치도|강원도|충청[남북]도|전라[남북]도'
        r'|전북특별자치도|경상[남북]도|제주특별자치도)'
        r'(?:\s+\S{1,10}[시군구]){1,2}\s+\S{1,15}(?:로|길|대로)\s+\d+[^\n|]{0,60})'
    )),
]

_US_STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO"
    "|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY"
)

_REGEX_EN = [
    ("SSN", re.compile(r'(?<!\d)(\d{3}\s*-\s*\d{2}\s*-\s*\d{4})(?!\d)')),
    # US phone: (415) 555-2389 | 415-555-2389 | 415.555.2389 | 1-800-555-2389
    ("PHONE", re.compile(r'(?<!\d)(\(?\d{3}\)?[-.\s]\s*\d{3}[-.\s]\d{4})(?!\d)')),
    ("PHONE", re.compile(r'(?<!\d)(1[-.\s]\d{3}[-.\s]\d{3}[-.\s]\d{4})(?!\d)')),
    ("EMAIL", re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')),
    ("CARD", re.compile(r'(?<!\d)(\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4})(?!\d)')),
    ("ACCOUNT", re.compile(r'(?<!\d)(\d{2,4}\s*-\s*\d{2,6}\s*-\s*\d{4,8})(?!\d)')),
    # US street address: number + street + ... + STATE ZIP
    ("ADDRESS", re.compile(
        r'(\d{1,6}\s+[A-Z][A-Za-z0-9.\' ]{2,40}'
        r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Park\sWest|Park)'
        r'[^\n]{0,60}?,\s*(?:' + _US_STATES + r')\s+\d{5}(?:-\d{4})?)'
    )),
]

REGEX_SETS = {"ko": _REGEX_KO, "en": _REGEX_EN}


# Regex types that are fixed-format IDs and should be dropped if they contain a
# masking pattern (``*``/``OO``/``00``). Natural-language types (ADDRESS/PERSON)
# may legitimately contain those characters, so they are excluded from the check.
FIXED_FORMAT_TYPES = ("RRN", "SSN", "ACCOUNT", "PHONE", "EMAIL", "CARD")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Bedrock analysis prompts.
# ─────────────────────────────────────────────────────────────────────────────

_ANALYSIS_PROMPTS_KO = {
    "insurance": "보험 청약서를 분석하여 심사 의견을 작성하세요. 건강 고지사항, 보험료 적정성, 보장 범위, 위험 요인을 평가하세요.",
    "mortgage": "주택담보대출 신청서를 분석하여 심사 의견을 작성하세요. DTI, LTV 비율, 소득 대비 상환 능력, 담보물 가치를 평가하세요.",
    "creditcard": "신용카드 발급 신청서를 분석하여 심사 의견을 작성하세요. 소득 대비 기존 카드 사용액, 신용도, 연회비 적정성을 평가하세요.",
    "stock": "증권 종합계좌 개설 신청서를 분석하여 심사 의견을 작성하세요. 투자 성향, 금융 자산, 적합성 원칙 준수 여부를 평가하세요.",
}

_ANALYSIS_PROMPTS_EN = {
    "insurance": "Analyze the insurance application and write an underwriting opinion. Evaluate the health disclosures, premium adequacy, coverage scope, and risk factors.",
    "mortgage": "Analyze the mortgage loan application and write an underwriting opinion. Evaluate the DTI and LTV ratios, repayment capacity relative to income, and collateral value.",
    "creditcard": "Analyze the credit card application and write an underwriting opinion. Evaluate existing card usage relative to income, creditworthiness, and annual-fee suitability.",
    "stock": "Analyze the brokerage account opening application and write a review opinion. Evaluate the investor's risk profile, financial assets, and compliance with the suitability principle.",
}

ANALYSIS_PROMPTS = {"ko": _ANALYSIS_PROMPTS_KO, "en": _ANALYSIS_PROMPTS_EN}


# The wrapper that frames the anonymized text for Bedrock. ``{scenario_prompt}`` and
# ``{anonymized_text}`` are filled in by the orchestrator. The instruction to keep PII
# tokens intact is what lets Stage 7 reassembly restore the original values.
_ANALYSIS_WRAPPER_KO = """다음은 익명화 처리된 금융 문서입니다. 개인정보는 [PERSON_1], [SSN_1] 등의 토큰으로 치환되어 있습니다.

{scenario_prompt}

한국어로 상세한 분석 리포트를 작성하세요. 주요 발견사항, 위험 요인, 최종 권고 의견을 포함하세요.
분석 결과에서 PII 토큰은 그대로 유지하세요 (예: [PERSON_1] 고객님).

=== 문서 내용 ===
{anonymized_text}"""

_ANALYSIS_WRAPPER_EN = """The following is an anonymized financial document. Personal information has been replaced with tokens such as [PERSON_1], [SSN_1].

{scenario_prompt}

Write a detailed analysis report in English. Include key findings, risk factors, and a final recommendation.
Keep the PII tokens unchanged in your output (e.g. "Dear [PERSON_1]").

=== Document ===
{anonymized_text}"""

ANALYSIS_WRAPPER = {"ko": _ANALYSIS_WRAPPER_KO, "en": _ANALYSIS_WRAPPER_EN}


def normalize_lang(lang: str | None) -> str:
    """Return a supported language code, defaulting to ``DEFAULT_LANG``."""
    if lang and lang.lower() in SUPPORTED_LANGS:
        return lang.lower()
    return DEFAULT_LANG
