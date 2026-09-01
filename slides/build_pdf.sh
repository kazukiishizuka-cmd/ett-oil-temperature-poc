#!/usr/bin/env bash
# 報告スライド（HTML）をPDFに変換する。
# Chromeのヘッドレス印刷を使う。CSSの @page でスライドの用紙サイズ（16:9）を決めている。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
HTML="${1:-$HERE/report.html}"
OUT="${2:-$HERE/PoC_report.pdf}"

if [ ! -x "$CHROME" ]; then
  echo "Chromeが見つからない: $CHROME" >&2
  echo "CHROME環境変数で実行ファイルのパスを指定してください" >&2
  exit 1
fi

"$CHROME" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=20000 \
  --print-to-pdf="$OUT" \
  "file://$HTML" 2>/dev/null

echo "生成: $OUT"
