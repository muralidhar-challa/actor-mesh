#!/usr/bin/env python3
"""
test_ner_c_compare.py — Compare C ONNX NER binary output against native spaCy.

Usage:
    python3 tools/test_ner_c_compare.py [model.onnx] [c_binary]

Defaults:
    model   = models/ner.onnx
    binary  = /tmp/test_ner_infer  (build with: see tools/test_ner_infer.c header)
"""
import sys
import subprocess
import spacy

MODEL_PATH  = sys.argv[1] if len(sys.argv) > 1 else "models/ner.onnx"
C_BINARY    = sys.argv[2] if len(sys.argv) > 2 else "/tmp/test_ner_infer"

nlp = spacy.load("en_core_web_sm")

TEXTS = [
    # original 6
    "Apple CEO Tim Cook met with Sundar Pichai in New York last Tuesday.",
    "Dr. Emily Chen joined Acme Corp on January 15th.",
    "The Federal Reserve raised rates by 25 basis points.",
    "Elon Musk founded SpaceX in Hawthorne, California in 2002.",
    "Barack Obama was born in Honolulu, Hawaii.",
    "Microsoft acquired LinkedIn for $26 billion in 2016.",
    # multi-token names / adjacency
    "Jeff Bezos stepped down as Amazon CEO in July 2021.",
    "Satya Nadella has led Microsoft since February 2014.",
    "Warren Buffett and Charlie Munger run Berkshire Hathaway.",
    "The United Nations headquarters is located in New York City.",
    "Angela Merkel served as Chancellor of Germany for 16 years.",
    "Cristiano Ronaldo plays for Al Nassr in Saudi Arabia.",
    # single-token (U-* path)
    "Tesla shares rose 5% on Friday.",
    "NASA launched the Artemis mission from Florida.",
    "Google was founded in Menlo Park.",
    # long, entity-dense
    "Kamala Harris met with Emmanuel Macron in Paris to discuss NATO and Ukraine.",
    "Amazon, Google, and Meta reported earnings on Thursday.",
    "The European Central Bank, led by Christine Lagarde, hiked rates in Frankfurt.",
    # ambiguous / tricky
    "Apple and Microsoft are both headquartered in the United States.",
    "The New York Times reported that President Biden signed the bill.",
    "Dr. Fauci advised the White House on the COVID-19 response.",
    # dates and money
    "SpaceX raised $2 billion in January 2023.",
    "The World Cup final was held in Lusail, Qatar on December 18, 2022.",
    # no entities
    "The quick brown fox jumps over the lazy dog.",
    "Interest rates remain unchanged after the committee vote.",
]

# ── parse C binary output ──────────────────────────────────────────────────────

def c_entities(text):
    """Run C binary, return list of (text_span, label) matching native spaCy format."""
    result = subprocess.run(
        [C_BINARY, MODEL_PATH, text],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [C binary error] {result.stderr.strip()}", flush=True)
        return None

    doc = nlp.make_doc(text)
    spans = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        start, end, label = int(parts[0]), int(parts[1]), parts[2]
        spans.append((doc[start:end].text, label))
    return spans

def native_entities(text):
    return [(e.text, e.label_) for e in nlp(text).ents]

# ── run ────────────────────────────────────────────────────────────────────────

col = 60
print(f"\n{'Text':<{col}}  C-vs-native")
print("─" * (col + 20))

all_match = True
mismatches = []

for text in TEXTS:
    native = native_entities(text)
    c_out  = c_entities(text)
    if c_out is None:
        print(f"{text[:col-1]:<{col}}  ERROR")
        all_match = False
        continue
    match = (native == c_out)
    all_match = all_match and match
    flag = "✓" if match else "≠"
    print(f"{text[:col-1]:<{col}}  {flag}")
    if not match:
        mismatches.append((text, native, c_out))
        print(f"  native: {native}")
        print(f"  C:      {c_out}")

print()
if all_match:
    print(f"ALL {len(TEXTS)} MATCH")
else:
    print(f"{len(mismatches)} / {len(TEXTS)} DIFFER")
