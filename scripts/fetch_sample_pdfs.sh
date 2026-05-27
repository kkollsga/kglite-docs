#!/usr/bin/env bash
# Download the sample open-access PDFs used in the test/demo suite.
# Run once after cloning. Skips files that already exist.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p sample_data/pdfs

declare -A pdfs=(
  [attention_is_all_you_need.pdf]="https://arxiv.org/pdf/1706.03762"
  [bert.pdf]="https://arxiv.org/pdf/1810.04805"
  [bge_m3.pdf]="https://arxiv.org/pdf/2402.03216"
  [colbert.pdf]="https://arxiv.org/pdf/2004.12832"
  [constitutional_ai.pdf]="https://arxiv.org/pdf/2212.08073"
  [dpo.pdf]="https://arxiv.org/pdf/2305.18290"
  [dpr.pdf]="https://arxiv.org/pdf/2004.04906"
  [gpt3.pdf]="https://arxiv.org/pdf/2005.14165"
  [llama2.pdf]="https://arxiv.org/pdf/2307.09288"
  [llama3.pdf]="https://arxiv.org/pdf/2407.21783"
  [lora.pdf]="https://arxiv.org/pdf/2106.09685"
  [mistral_7b.pdf]="https://arxiv.org/pdf/2310.06825"
  [rag.pdf]="https://arxiv.org/pdf/2005.11401"
  [sbert.pdf]="https://arxiv.org/pdf/1908.10084"
  [nist_ai_rmf.pdf]="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
  [irs_p15.pdf]="https://www.irs.gov/pub/irs-pdf/p15.pdf"
)

for name in "${!pdfs[@]}"; do
  out="sample_data/pdfs/$name"
  if [[ -s "$out" ]]; then
    echo "✓ $name (already present)"
    continue
  fi
  url="${pdfs[$name]}"
  echo "→ downloading $name from $url"
  curl -sSL "$url" -o "$out"
  if [[ ! -s "$out" ]]; then
    echo "  ✗ empty file, removing"
    rm -f "$out"
  fi
done

echo
echo "Done. PDFs in sample_data/pdfs/"
ls -la sample_data/pdfs/
