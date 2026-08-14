#!/usr/bin/env node
/**
 * Generate a PDF for each sample-documents/<lang>/<scenario>.txt, so the OCR
 * path (Stage 2, ocr-service) has something to actually parse end to end.
 *
 * The .txt files remain the source of truth — this script only renders them.
 * Layout is intentionally plain (title + monospace body, preserving the
 * original line breaks) rather than a realistic form mockup: that keeps the
 * OCR'd text close to the .txt content (useful for spot-checking OCR output
 * against it) and keeps the embedded font subset tiny, which is what keeps
 * each PDF under ~100KB — a full weight of Noto Sans KR embedded unsubsetted
 * (as opposed to per-page glyph subsetting) is several MB on its own.
 *
 * Requires Playwright's Chromium (not a runtime dependency of the pipeline —
 * only needed to regenerate these PDFs):
 *   npx playwright install chromium
 *   node scripts/generate_sample_pdfs.mjs
 *
 * Rendering Korean text needs a CJK-capable font available to the browser
 * (e.g. `fonts-noto-cjk` on Debian/Ubuntu) — without one, Hangul renders as
 * blank boxes. Most users will never need to run this: the generated PDFs
 * are committed under sample-documents/.
 */
import { chromium } from "playwright";
import { readFileSync, writeFileSync, readdirSync } from "fs";
import { resolve, dirname, basename } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");
const docsRoot = resolve(projectRoot, "sample-documents");

const TITLES = {
  insurance: { ko: "보험 청약서", en: "Insurance Application" },
  mortgage: { ko: "주택담보대출 신청서", en: "Mortgage Loan Application" },
  creditcard: { ko: "신용카드 발급 신청서", en: "Credit Card Application" },
  stock: { ko: "증권 종합계좌 개설 신청서", en: "Brokerage Account Opening Application" },
};

function htmlFor(lang, scenario, bodyText) {
  const title = TITLES[scenario]?.[lang] ?? scenario;
  // Body text already has its own title line as the first line — drop it to
  // avoid printing the title twice with slightly different styling.
  const bodyWithoutTitle = bodyText.split("\n").slice(1).join("\n").trim();
  const escaped = bodyWithoutTitle
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return `<!DOCTYPE html>
<html lang="${lang}">
<head>
<meta charset="UTF-8">
<style>
  body {
    font-family: ${lang === "ko" ? "'Noto Sans CJK KR', 'Noto Sans KR', sans-serif" : "Arial, sans-serif"};
    font-size: 12px;
    line-height: 1.6;
    color: #111;
    padding: 32px 40px;
  }
  h1 { font-size: 18px; border-bottom: 2px solid #333; padding-bottom: 8px; margin-bottom: 16px; }
  pre { white-space: pre-wrap; font-family: inherit; margin: 0; }
</style>
</head>
<body>
<h1>${title}</h1>
<pre>${escaped}</pre>
</body>
</html>`;
}

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  for (const lang of ["ko", "en"]) {
    const langDir = resolve(docsRoot, lang);
    const files = readdirSync(langDir).filter((f) => f.endsWith(".txt"));
    for (const file of files) {
      const scenario = basename(file, ".txt");
      const text = readFileSync(resolve(langDir, file), "utf-8");
      await page.setContent(htmlFor(lang, scenario, text), { waitUntil: "networkidle" });
      const pdfPath = resolve(langDir, `${scenario}.pdf`);
      await page.pdf({ path: pdfPath, format: "A4", margin: { top: "0", bottom: "0", left: "0", right: "0" } });
      console.log(`wrote ${pdfPath}`);
    }
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
