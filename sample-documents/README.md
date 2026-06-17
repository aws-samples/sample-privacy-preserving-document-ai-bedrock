# Sample Documents

These files represent the **text output of an OCR step** on financial application
forms. This sample assumes OCR has already been performed — it is the input to the
privacy-preserving pipeline (PII detection → masking → analysis → reassembly).

Two language sets are provided. Select one with the pipeline's `lang` parameter:

| Path | Language | National ID format | Selected by |
|------|----------|--------------------|-------------|
| `ko/` | Korean | RRN (`850315-1234567`) | `lang=ko` |
| `en/` | English (US) | SSN (`521-84-6390`) | `lang=en` |

Each set has four scenarios:

- `insurance.txt` — insurance underwriting application
- `mortgage.txt` — mortgage loan application
- `creditcard.txt` — credit card issuance application
- `stock.txt` — brokerage account opening application

> **All names, ID numbers, account numbers, addresses, and financial figures are
> entirely fictitious** and were generated for demonstration purposes only. Any
> resemblance to real persons or accounts is coincidental.

To add another language, drop a `<lang>/<scenario>.txt` set here and register a matching
detection prompt, regex set, and analysis prompt in `src/orchestrator/prompts.py`.
