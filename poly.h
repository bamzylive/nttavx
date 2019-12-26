#pragma once

#include <stdint.h>

void poly_tobyte(unsigned char *a, uint16_t *r);

void poly_tobyte_avx(unsigned char *a, uint16_t *r);

void byte_topoly(unsigned char *a, uint16_t *r);

void byte_topoly_avx(unsigned char *a, uint16_t *r);

void poly_add(uint16_t *c, const uint16_t *a, const uint16_t *b);

void poly_add_avx(uint16_t *c, const uint16_t *a, const uint16_t *b);

// void poly_gen_a(uint16_t *a, const unsigned char *seed);
//
// void poly_gen_a_avx(uint16_t *a, const unsigned char *seed);

// void poly_sample(uint16_t *r, unsigned char * seed, unsigned char nonce);
//
// void poly_sample_avx(uint16_t *r, unsigned char * seed, unsigned char nonce);

int verify(const unsigned char *a, const unsigned char *b, size_t len);
int verify_avx(const unsigned char *a, const unsigned char *b, size_t len);

void cmov(unsigned char *r, const unsigned char *x, size_t len,
          unsigned char b);

int test_poly();

int test_CBD();