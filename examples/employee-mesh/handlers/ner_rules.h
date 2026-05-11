/* ner_rules.h — Regex-based PII detection as post-process layer.
 * 
 * Usage:
 *   PiiSpan spans[128];
 *   int n = pii_rules_scan(text, spans, 128);
 *   // spans now contains email/phone/SSN/CC matches
 *   // Merge with NER model output for complete PII coverage
 */
#ifndef NER_RULES_H
#define NER_RULES_H

#include <regex.h>
#include <string.h>

#define PII_MAX_SPANS 128
#define PII_MAX_PATTERNS 8

typedef struct {
    int start;
    int end;
    char label[16];  // EMAIL, PHONE, SSN, CC, IP, etc.
} PiiSpan;

/* Built-in PII regex patterns */
static const char* PII_PATTERNS[][2] = {
    {"EMAIL",       "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"},
    {"PHONE",       "\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b"},
    {"SSN",         "\\b\\d{3}-\\d{2}-\\d{4}\\b"},
    {"CREDIT_CARD", "\\b\\d{4}[ -]?\\d{4}[ -]?\\d{4}[ -]?\\d{4}\\b"},
    {"IP_ADDRESS",  "\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b"},
    {"URL",         "https?://[^\\s]+"},
};

static regex_t g_pii_regex[PII_MAX_PATTERNS];
static int g_pii_compiled = 0;

/* One-time compilation of all PII regex patterns */
static inline void pii_rules_init(void) {
    if (g_pii_compiled) return;
    for (int i = 0; i < 6; i++) {
        if (regcomp(&g_pii_regex[i], PII_PATTERNS[i][1], REG_EXTENDED | REG_ICASE)) {
            fprintf(stderr, "pii_rules: failed to compile %s\n", PII_PATTERNS[i][0]);
        }
    }
    g_pii_compiled = 1;
}

/* Scan text for PII patterns. Returns number of spans found.
 * spans_out must be pre-allocated with max_spans entries.
 * Merge with NER output: patterns override ML for these types. */
static inline int pii_rules_scan(const char* text, PiiSpan* spans_out, int max_spans) {
    if (!g_pii_compiled) pii_rules_init();
    
    int n_spans = 0;
    int text_len = (int)strlen(text);
    
    for (int p = 0; p < 6 && n_spans < max_spans; p++) {
        regmatch_t match;
        int offset = 0;
        
        while (offset < text_len && n_spans < max_spans) {
            if (regexec(&g_pii_regex[p], text + offset, 1, &match, 0) == 0) {
                spans_out[n_spans].start = offset + (int)match.rm_so;
                spans_out[n_spans].end   = offset + (int)match.rm_eo;
                strncpy(spans_out[n_spans].label, PII_PATTERNS[p][0], 15);
                spans_out[n_spans].label[15] = '\0';
                n_spans++;
                offset += (int)match.rm_eo;
            } else {
                break;
            }
        }
    }
    return n_spans;
}

/* Check if a text position falls within a rule-based span.
 * Used to skip NER processing for tokens already caught by rules. */
static inline const char* pii_rules_lookup(const PiiSpan* spans, int n_spans, int pos) {
    for (int i = 0; i < n_spans; i++) {
        if (pos >= spans[i].start && pos < spans[i].end) {
            return spans[i].label;
        }
    }
    return NULL;
}

#endif /* NER_RULES_H */
