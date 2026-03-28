#!/usr/bin/env bash
# Pull required Ollama models and verify they are available.
# Run this once after starting docker-compose.

set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

pull() {
    echo "→ Pulling $1 ..."
    curl -s -X POST "${OLLAMA_URL}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$1\"}" | \
        python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if 'status' in d:
        print('  ', d['status'], d.get('digest', ''))
    if d.get('status') == 'success':
        print('  ✓ Done')
        break
"
}

echo "============================================"
echo " HA Agent – Ollama model setup"
echo "============================================"
echo ""

# Tool-calling model (4B, fast)
# nemotron-mini is NVIDIA's 4B instruction/tool model
pull "nemotron-mini"

echo ""

# General conversation model
# Qwen 2.5 7B is a strong multilingual model (EN + DE)
pull "qwen2.5:7b"

echo ""
echo "============================================"
echo " Models installed. Warming up..."
echo "============================================"

for model in nemotron-mini qwen2.5:7b; do
    echo "→ Warmup $model"
    curl -s -X POST "${OLLAMA_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${model}\", \"prompt\": \"\", \"keep_alive\": -1}" > /dev/null
    echo "  ✓ $model loaded"
done

echo ""
echo "Done! Models are hot and ready."
