/*
 * tools/test_ner_infer.c — end-to-end NER: tokenise → features → ONNX → decode
 *
 * Build (from repo root):
 *   gcc -O2 -std=c11 -I. \
 *       tools/test_ner_infer.c \
 *       vendor/smhasher/MurmurHash2.cpp \
 *       -lstdc++ -lonnxruntime \
 *       -o /tmp/test_ner_infer
 *
 * Run:
 *   /tmp/test_ner_infer models/ner.onnx \
 *       "Apple CEO Tim Cook met with Sundar Pichai in New York last Tuesday."
 *
 * stdout: one entity per line — "start end label"
 * stderr: diagnostics (suppress with 2>/dev/null)
 */
#include <stdio.h>
#include <string.h>
#include "examples/employee-mesh/handlers/ner_infer.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.onnx> <text>\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *text       = argv[2];

    NerModel *m = ner_model_load(model_path);
    fprintf(stderr, "seeds: NORM=%u PREFIX=%u SUFFIX=%u SHAPE=%u\n",
            m->seeds.norm_seed, m->seeds.prefix_seed,
            m->seeds.suffix_seed, m->seeds.shape_seed);

    NerToken toks[NER_MAX_TOKENS];
    int n = spacy_tokenize(text, toks, NER_MAX_TOKENS);
    fprintf(stderr, "tokens (%d):", n);
    for (int i = 0; i < n; i++) fprintf(stderr, " [%s]", toks[i].text);
    fprintf(stderr, "\n");

    uint64_t feats[NER_MAX_TOKENS * NER_N_FEATS];
    ner_build_feature_matrix(toks, n, &m->seeds, feats);

    float table[(NER_MAX_TOKENS + 1) * NER_N_FEATS_PA * NER_N_HIDDEN * NER_N_PIECES];
    ner_model_run(m, feats, n, table);

    NerSpan spans[NER_MAX_TOKENS];
    int ns = ner_decode_spans(table, &m->dec, n, spans);

    for (int i = 0; i < ns; i++)
        printf("%d %d %s\n", spans[i].start, spans[i].end, spans[i].label);

    ner_model_free(m);
    return 0;
}
