#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <windows.h>

#include "cbd.h"
#include "parameter.h"

void CBD_avx(uint16_t *a, unsigned char *bytes) {
  __m256i mask0 = _mm256_set1_epi32(0x3030303);

  __m256i mask_high = _mm256_set1_epi32(0xff00ff00);
  __m256i mask_low = _mm256_set1_epi32(0x00ff00ff);

  // the high byte and low byte of {0, 2, Q-2};
  __m256i a_low = _mm256_set_epi8(
      0x00, 0x02, 0xff, 0x00, 0x00, 0x02, 0xff, 0x00, 0x00, 0x02, 0xff, 0x00,
      0x00, 0x02, 0xff, 0x00, 0x00, 0x02, 0xff, 0x00, 0x00, 0x02, 0xff, 0x00,
      0x00, 0x02, 0xff, 0x00, 0x00, 0x02, 0xff, 0x00);
  __m256i a_high = _mm256_set_epi8(
      0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00,
      0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00,
      0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00);

  // the high byte and low byte of {0, 2Q+2, 2Q-2};
  //	__m256i a_low = _mm256_set_epi8(0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
  //0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
  //0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
  //0x00, 0x00);
  //	__m256i a_high = _mm256_set_epi8(0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  //0x1a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  //0x1a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  //0x1a, 0x00);

  __m256i y0, y1, y2, y3, y4, y5, temp;

  int i;
  for (i = 0; i < N / 128; i++) {
    y5 = _mm256_loadu_si256((__m256i *)(bytes + i * 32));
    y0 = _mm256_and_si256(y5, mask0);

    y1 = _mm256_shuffle_epi8(a_high, y0);
    y2 = _mm256_shuffle_epi8(a_low, y0);

    // temp = _mm256_unpackhi_epi8(y2, y1);
    //_mm256_storeu_si256((__m256i*)(a + 16 + 128 * i), temp);

    y3 = _mm256_srli_epi16(y2, 8);
    temp = _mm256_and_si256(y1, mask_high);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 16 + 128 * i), temp);

    y3 = _mm256_slli_epi16(y1, 8);
    temp = _mm256_and_si256(y2, mask_low);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 128 * i), temp);

    // take the next 2 bits
    y5 = _mm256_srli_epi16(y5, 2);
    y0 = _mm256_and_si256(y5, mask0);

    // the same procedure as the first step
    y1 = _mm256_shuffle_epi8(a_high, y0);
    y2 = _mm256_shuffle_epi8(a_low, y0);

    y3 = _mm256_srli_epi16(y2, 8);
    temp = _mm256_and_si256(y1, mask_high);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 48 + 128 * i), temp);

    y3 = _mm256_slli_epi16(y1, 8);
    temp = _mm256_and_si256(y2, mask_low);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 32 + 128 * i), temp);

    y5 = _mm256_srli_epi16(y5, 2);
    y0 = _mm256_and_si256(y5, mask0);

    y1 = _mm256_shuffle_epi8(a_high, y0);
    y2 = _mm256_shuffle_epi8(a_low, y0);

    y3 = _mm256_srli_epi16(y2, 8);
    temp = _mm256_and_si256(y1, mask_high);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 80 + 128 * i), temp);

    y3 = _mm256_slli_epi16(y1, 8);
    temp = _mm256_and_si256(y2, mask_low);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 64 + 128 * i), temp);

    y5 = _mm256_srli_epi16(y5, 2);
    y0 = _mm256_and_si256(y5, mask0);

    y1 = _mm256_shuffle_epi8(a_high, y0);
    y2 = _mm256_shuffle_epi8(a_low, y0);

    y3 = _mm256_srli_epi16(y2, 8);
    temp = _mm256_and_si256(y1, mask_high);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 112 + 128 * i), temp);

    y3 = _mm256_slli_epi16(y1, 8);
    temp = _mm256_and_si256(y2, mask_low);
    temp = _mm256_xor_si256(y3, temp);
    _mm256_storeu_si256((__m256i *)(a + 96 + 128 * i), temp);
  }
}

// 256 bytes generates 1024 coeffs
void CBD_ref(uint16_t *r, unsigned char *buf) {
  int i, iq, ir, j, k;
  static const uint16_t x[4] = {Q * 2, (Q - 1) * 2, (Q + 1) * 2, Q * 2};
  for (k = 0; k < 8; k++) {
    for (i = 0; i < 16; i++) {
      j = buf[i * 2 + k * 32];
      r[i + k * 128] = x[j & 0x3];
      r[i + 32 + k * 128] = x[(j >> 2) & 0x3];
      r[i + 64 + k * 128] = x[(j >> 4) & 0x3];
      r[i + 96 + k * 128] = x[(j >> 6) & 0x3];

      j = buf[i * 2 + 1 + k * 32];
      r[i + 16 + k * 128] = x[j & 0x3];
      r[i + 48 + k * 128] = x[(j >> 2) & 0x3];
      r[i + 80 + k * 128] = x[(j >> 4) & 0x3];
      r[i + 112 + k * 128] = x[(j >> 6) & 0x3];
    }
  }
}

int test_CBD() {
  LARGE_INTEGER start, end;
  LARGE_INTEGER frequency;

  uint16_t a[1024], b[1024];
  unsigned char bytes[256];

  int i, j;
  double elapsed;

  QueryPerformanceFrequency(&frequency);

  double quadpart = (double)frequency.QuadPart;

  for (i = 0; i < 256; i++) {
    bytes[i] = (i * 29) & 0xff;
    //		bytes[i] = 0;
  }

  QueryPerformanceCounter(&start);
  for (j = 0; j < 1000000; j++) {
    CBD_avx(a, bytes);
  }

  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;
  printf("CBD_AVX RUNS in Time : %0.6f usecs\n ", elapsed);

  QueryPerformanceCounter(&start);
  for (j = 0; j < 1000000; j++)
    CBD_ref(b, bytes);
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;
  printf("CBD_REF RUNS in Time : %0.6f usecs\n ", elapsed);

  for (i = 0; i < 1024; i++) {
    if (a[i] % 3329 != b[i] % 3329) {
      return -1;
    }
  }
  //
  //
  //	/*TEST For the arrangement of __m256i */
  //	for (i = 0; i < 16; i++)
  //	{
  //		a[i] = 0xffff;
  //		b[i] = 2;
  //	}
  //
  //	__m256i y5 = _mm256_loadu_si256((__m256i*)(a));
  //	__m256i y6 = _mm256_loadu_si256((__m256i*)(b));
  ////	y5 = _mm256_slli_si256(y5, 1);
  ////	y5 = _mm256_slli_epi32(y5, 1);
  ////	__m256i y7 = _mm256_mulhi_epu16(y5, y6);
  ////	__m256i y8 = _mm256_mullo_epi16(y5, y6);
  //
  //
  //	__m256i y7 = _mm256_subs_epu16(y5, y6);
  //	_mm256_storeu_si256((__m256i*)(a), y7);
  ////	_mm256_storeu_si256((__m256i*)(b), y8);
  //
  //	for (i = 0; i < 16; i++)
  //	{
  //		printf("%04x, ", a[i]);
  //	}
  //	printf("\n\n");
  //
  //	//for (i = 0; i < 16; i++)
  //	//{
  //	//	printf("%04x, ", b[i]);
  //	//}
  //	//printf("\n");
  //
  //	y7 = _mm256_adds_epi16(y5, y6);
  //
  //	_mm256_storeu_si256((__m256i*)(a), y7);
  ////	_mm256_storeu_si256((__m256i*)(b), y8);
  //
  //	for (i = 0; i < 16; i++)
  //	{
  //		printf("%04x, ", a[i]);
  //	}
  //	printf("\n\n");
  //	return 0;
}