#pragma once

#include <stdint.h>

// generate central binomial distributed coefficients of polynomial a, using
// random bytes of length
void CBD_avx(uint16_t *a, unsigned char *bytes);

// the original  poly_sample();
void CBD_ref(uint16_t *r, unsigned char *buf);

int test_CBD();