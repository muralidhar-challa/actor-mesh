/*
 * tools/test_ner_infer.c — end-to-end smoke test: tokenise → features → ONNX
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

    /* load model */
    NerModel *m = ner_model_load(model_path);
    printf("Model loaded. Seeds: NORM=%u PREFIX=%u SUFFIX=%u SHAPE=%u\n",
           m->seeds.norm_seed, m->seeds.prefix_seed,
           m->seeds.suffix_seed, m->seeds.shape_seed);

    /* tokenise */
    NerToken toks[NER_MAX_TOKENS];
    int n = spacy_tokenize(text, toks, NER_MAX_TOKENS);
    printf("Tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" [%s]", toks[i].text);
    printf("\n");

    /* feature matrix */
    uint64_t feats[NER_MAX_TOKENS * NER_N_FEATS];
    ner_build_feature_matrix(toks, n, &m->seeds, feats);

    /* inference */
    float vecs[NER_MAX_TOKENS * 64];
    ner_model_run(m, feats, n, vecs);

    /* print first 4 dims of each token vector */
    printf("Token vectors (first 4 dims):\n");
    for (int i = 0; i < n; i++) {
        printf("  [%2d] %-12s  %.4f %.4f %.4f %.4f\n",
               i, toks[i].text,
               vecs[i*64+0], vecs[i*64+1], vecs[i*64+2], vecs[i*64+3]);
    }

    ner_model_free(m);
    return 0;
}
