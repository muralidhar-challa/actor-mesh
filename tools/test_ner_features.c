/*
 * tools/test_ner_features.c — verify ner_features.h against spaCy output
 *
 * Build:
 *   gcc -O2 -std=c11 -I.. \
 *       tools/test_ner_features.c \
 *       vendor/smhasher/MurmurHash2.cpp vendor/smhasher/MurmurHash3.cpp \
 *       -lstdc++ -o /tmp/test_ner_features
 *
 * (or use g++ directly since smhasher sources are C++)
 *
 * Run:
 *   /tmp/test_ner_features
 *
 * Expected output is printed as Python-comparable values.
 * Cross-check with:
 *   python3 tools/test_ner_features.py
 */
#include <stdio.h>
#include <string.h>
#include "examples/employee-mesh/handlers/ner_features.h"

static void print_row(const char *label, const uint64_t *row)
{
    printf("  %-12s  ", label);
    for (int i = 0; i < NER_N_FEATS; i++)
        printf("%20llu%s", (unsigned long long)row[i], i < NER_N_FEATS-1 ? " " : "\n");
}

int main(void)
{
    const char *texts[] = {
        "Apple CEO Tim Cook met with Sundar Pichai in New York last Tuesday.",
        "Dr. Emily Chen joined Acme Corp on 2024-01-15.",
        "hello world",
    };
    int n_texts = sizeof(texts) / sizeof(texts[0]);

    NerSeeds seeds = NER_DEFAULT_SEEDS;

    for (int t = 0; t < n_texts; t++) {
        NerToken toks[NER_MAX_TOKENS];
        int n = spacy_tokenize(texts[t], toks, NER_MAX_TOKENS);

        printf("\nText: %s\n", texts[t]);
        printf("Tokens (%d):", n);
        for (int i = 0; i < n; i++) printf(" [%s]", toks[i].text);
        printf("\n");

        uint64_t feats[NER_MAX_TOKENS * NER_N_FEATS];
        ner_build_feature_matrix(toks, n, &seeds, feats);

        printf("Feature matrix [tok_idx: NORM×4 PREFIX×4 SUFFIX×4 SHAPE×4]:\n");
        for (int i = 0; i < n; i++) {
            printf("  tok[%2d] %-12s  ", i, toks[i].text);
            for (int j = 0; j < NER_N_FEATS; j++)
                printf("%llu%s", (unsigned long long)feats[i * NER_N_FEATS + j],
                       j < NER_N_FEATS-1 ? "," : "\n");
        }
    }

    /* quick sanity: MurmurHash64A("hello", 5, 0) should be stable across runs */
    uint64_t h = spacy_hash_string("hello", 5);
    printf("\nMurmurHash64A(\"hello\", 5, 0) = %llu\n", (unsigned long long)h);

    uint64_t h4[4];
    thinc_hash4(h, 8, h4);
    printf("thinc_hash4(hello_id, seed=8) = %llu,%llu,%llu,%llu\n",
           (unsigned long long)h4[0], (unsigned long long)h4[1],
           (unsigned long long)h4[2], (unsigned long long)h4[3]);

    return 0;
}
