#pragma once
#include <stdint.h>

uint16_t montgomery_reduce(uint32_t a);

uint16_t barrett_reduce(uint16_t a);

void poly_ntt(uint16_t *p);

void poly_ntt_avx(uint16_t *p);

void poly_invntt(uint16_t *a);

void poly_invntt_avx(uint16_t *a);

void poly_pointwise(uint16_t *h, uint16_t *f, uint16_t *g);

void poly_pointwise_avx(uint16_t *h, uint16_t *f, uint16_t *g);

void kntt(uint16_t *r);

void kinv_ntt(uint16_t *r);

void poly_invntt_avx(uint16_t *a);

void kntt_avx(uint16_t *r);

void kinv_ntt_avx(uint16_t *r);

int test_ntt();
