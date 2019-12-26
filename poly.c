#include <immintrin.h>
#include <stdint.h>
#include <time.h>
#include <windows.h>

#include "parameter.h"
#include "poly.h"

#define qinv 3327
#define BCON 20159

#define test_time 0

#define test_correctness 1

void poly_tobyte(unsigned char *a, uint16_t *r) {
  int i, temp, j, k;
#if test_time
  uint16_t t_0, t_1;
  temp = 0;
  for (i = 0; i < (N >> 1); i++) {
    t_0 = r[(i << 1) + 0];
    t_1 = r[(i << 1) + 1];
    a[temp] = t_0 & 0xff;
    a[temp + 1] = (t_0 >> 8) | (t_1 << 4) & 0xff;
    a[temp + 2] = (t_1 >> 4) & 0xff;
    temp = temp + 3;
  }
#elif test_correctness
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      a[i * 96 + j * 2 + 1] =
          ((r[64 * i + 48 + j] & 0x0f00) >> 4) ^ (r[64 * i + j] >> 8);
      a[i * 96 + j * 2] = (r[64 * i + j] & 0xff);
    }
    for (j = 0; j < 16; j++) {
      a[i * 96 + 32 + j * 2 + 1] =
          ((r[64 * i + 48 + j] & 0x00f0)) ^ (r[64 * i + 16 + j] >> 8);
      a[i * 96 + 32 + j * 2] = (r[64 * i + 16 + j] & 0xff);
    }

    for (j = 0; j < 16; j++) {
      a[i * 96 + 64 + j * 2 + 1] =
          ((r[64 * i + 48 + j] & 0x000f) << 4) ^ (r[64 * i + 32 + j] >> 8);
      a[i * 96 + 64 + j * 2] = (r[64 * i + 32 + j] & 0xff);
    }
  }
#endif
}

void poly_tobyte_avx(unsigned char *a, uint16_t *r) {
  __m256i temp[8];
  __m256i mask = _mm256_set1_epi16(0xf000);
  int i;
  // use 4 256-bit registers to hold 64 coefficients, while all 768 valid bits
  // fits into 3 256-bit registers, hence 16 loops
  for (i = 0; i < 16; i++) {
    temp[0] = _mm256_loadu_si256((__m256i *)(r + 64 * i));
    temp[1] = _mm256_loadu_si256((__m256i *)(r + 64 * i + 16));
    temp[2] = _mm256_loadu_si256((__m256i *)(r + 64 * i + 32));
    temp[3] = _mm256_loadu_si256((__m256i *)(r + 64 * i + 48));

    temp[4] = _mm256_slli_epi16(temp[3], 4);
    temp[4] = _mm256_and_si256(temp[4], mask);
    temp[0] = _mm256_xor_si256(temp[4], temp[0]);
    _mm256_storeu_si256((__m256i *)(a + 96 * i), temp[0]);

    temp[4] = _mm256_slli_epi16(temp[3], 8);
    temp[4] = _mm256_and_si256(temp[4], mask);
    temp[1] = _mm256_xor_si256(temp[4], temp[1]);
    _mm256_storeu_si256((__m256i *)(a + 96 * i + 32), temp[1]);

    temp[4] = _mm256_slli_epi16(temp[3], 12);
    temp[4] = _mm256_and_si256(temp[4], mask);
    temp[2] = _mm256_xor_si256(temp[4], temp[2]);
    _mm256_storeu_si256((__m256i *)(a + 96 * i + 64), temp[2]);
  }
}

void byte_topoly(unsigned char *a, uint16_t *r) {
  int i, j;
  uint16_t t0, t1;
#if test_time
  for (i = 0; i < (N >> 1); i++) {
    t0 = i << 1;
    t1 = t0 + i;
    r[t0] = a[t1] | (((uint16_t)a[t1 + 1] & 0xf) << 8);
    r[t0 + 1] = a[t1 + 1] >> 4 | (((uint16_t)a[t1 + 2] & 0xff) << 4);
  }

#elif test_correctness
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {

      r[64 * i + j] =
          ((uint16_t)(a[i * 96 + j * 2 + 1] & 0xf) << 8) ^ a[i * 96 + j * 2];
      r[64 * i + 48 + j] = a[i * 96 + j * 2 + 1] & 0xf0;
    }
    for (j = 0; j < 16; j++) {
      r[64 * i + 16 + j] = ((uint16_t)(a[i * 96 + 32 + j * 2 + 1] & 0xf) << 8) ^
                           a[i * 96 + 32 + j * 2];

      r[64 * i + 48 + j] =
          (r[64 * i + 48 + j] << 4) ^ (a[i * 96 + 32 + j * 2 + 1] & 0xf0);
    }

    for (j = 0; j < 16; j++) {
      r[64 * i + 32 + j] = ((uint16_t)(a[i * 96 + 64 + j * 2 + 1] & 0xf) << 8) ^
                           a[i * 96 + 64 + j * 2];

      r[64 * i + 48 + j] =
          (r[64 * i + 48 + j]) ^ (a[i * 96 + 64 + j * 2 + 1] >> 4);
    }
  }
#endif
}

void byte_topoly_avx(unsigned char *a, uint16_t *r) {
  __m256i temp[8];
  __m256i mask = _mm256_set1_epi16(0x0fff);
  __m256i mask_hi = _mm256_set1_epi16(0xf000);
  int i;
  // use 3 256-bit registers to hold  768 valid bits, and  convert into 768/12 =
  // 64 coefficients, which fits into 4 256-bit registers
  for (i = 0; i < 16; i++) {
    temp[0] = _mm256_loadu_si256((__m256i *)(a + 96 * i));
    temp[1] = _mm256_loadu_si256((__m256i *)(a + 96 * i + 32));
    temp[2] = _mm256_loadu_si256((__m256i *)(a + 96 * i + 64));

    temp[3] = _mm256_and_si256(temp[0], mask);
    _mm256_storeu_si256((__m256i *)(r + 64 * i), temp[3]);

    temp[3] = _mm256_and_si256(temp[1], mask);
    _mm256_storeu_si256((__m256i *)(r + 64 * i + 16), temp[3]);

    temp[3] = _mm256_and_si256(temp[2], mask);
    _mm256_storeu_si256((__m256i *)(r + 64 * i + 32), temp[3]);

    temp[0] = _mm256_and_si256(temp[0], mask_hi);
    temp[0] = _mm256_srli_epi16(temp[0], 4);

    temp[1] = _mm256_and_si256(temp[1], mask_hi);
    temp[1] = _mm256_srli_epi16(temp[1], 8);

    temp[2] = _mm256_srli_epi16(temp[2], 12);

    temp[3] = _mm256_xor_si256(temp[0], temp[1]);
    temp[3] = _mm256_xor_si256(temp[3], temp[2]);

    _mm256_storeu_si256((__m256i *)(r + 64 * i + 48), temp[3]);
  }
}

void poly_add(uint16_t *c, const uint16_t *a, const uint16_t *b) {
  int i;
  for (i = 0; i < N; i++)
    c[i] = (a[i] + b[i]) % Q;
}

void poly_add_avx(uint16_t *c, const uint16_t *a, const uint16_t *b) {
  __m256i A, B, C, temp;
  __m256i xQ_avx = _mm256_set1_epi16(Q);
  int i;
  for (i = 0; i < 64; i++) {
    A = _mm256_loadu_si256((__m256i *)(a + 16 * i));
    B = _mm256_loadu_si256((__m256i *)(b + 16 * i));
    C = _mm256_add_epi16(A, B);

    C = _mm256_sub_epi16(C, xQ_avx);

    temp = _mm256_srai_epi16(C, 16);
    temp = _mm256_and_si256(temp, xQ_avx);
    C = _mm256_add_epi16(C, temp);
    _mm256_storeu_si256((__m256i *)(c + 16 * i), C);
  }
}

// void poly_gen_a(uint16_t *a, const unsigned char *seed)
//{
//        unsigned int ctr=0;
//        uint16_t val;
//        uint64_t state[25];
//        unsigned char buf[SHAKE128_RATE];
//        unsigned char extseed[PKE_SEED_BYTES+1];
//        int i,j;
//
//        for(i=0;i<PKE_SEED_BYTES;i++)
//            extseed[i] = seed[i];
//
//        for(i=0;i<(N>>6);i++) /* generate a in blocks of 64 coefficients */
//        {
//            ctr = 0;
//            extseed[PKE_SEED_BYTES] = i; /* domain-separate the 16 independent
//            calls */ shake128_absorb(state, extseed, PKE_SEED_BYTES+1);
//            while(ctr<64) /* Very unlikely to run more than once */
//            {
//                shake128_squeezeblocks(buf,1,state);
//                for(j=0;j<SHAKE128_RATE && ctr < 64;j+=2)
//                {
//                    val = (buf[j] | ((uint16_t) buf[j+1] << 8));
//                    if(val<19*Q)
//                    {
//
//
//                       //val-= (val>>12)*Q;
//					   a[i*64+ctr]=barrett_reduce(val);
//                        ctr++;
//                    }
//                }
//            }
//        }
//
//}
//
//
// void poly_gen_a_avx(uint16_t *a, const unsigned char *seed)
//{
//	unsigned int ctr = 0;
//	uint16_t val;
//	uint64_t state[25];
//	unsigned char buf[SHAKE128_RATE];
//	unsigned char extseed[PKE_SEED_BYTES + 1];
//	int i, j;
//
//	for (i = 0; i < PKE_SEED_BYTES; i++)
//		extseed[i] = seed[i];
//
//	for (i = 0; i < (N >> 6); i++) /* generate a in blocks of 64
//coefficients */
//	{
//		ctr = 0;
//		extseed[PKE_SEED_BYTES] = i; /* domain-separate the 16 independent
//calls */ 		shake128_absorb(state, extseed, PKE_SEED_BYTES + 1); 		while (ctr < 64)
///* Very unlikely to run more than once */
//		{
//			shake128_squeezeblocks(buf, 1, state);
//			for (j = 0; j < SHAKE128_RATE && ctr < 64; j += 2)
//			{
//				val = (buf[j] | ((uint16_t)buf[j + 1] << 8));
//				if (val < 19 * Q)
//				{
//
//
//					//val-= (val>>12)*Q;
//					a[i * 64 + ctr] = barrett_reduce(val);
//					ctr++;
//				}
//			}
//		}
//	}
//
//}

// void poly_sample(uint16_t *r, unsigned char * seed, unsigned char nonce)
//
//{
//	int i, j, iq, ir;
//	unsigned char extseed[PKE_SEED_BYTES + 1], buf[N4];
//	for (i = 0; i < PKE_SEED_BYTES; i++)
//		extseed[i] = seed[i];
//	extseed[PKE_SEED_BYTES] = nonce;
//	shake256(buf, N4, extseed, PKE_SEED_BYTES + 1);
//
//	for (i = 0; i < N; i++)
//	{
//		iq = (i >> 3);
//		ir = i & 7;
//		r[i] = (((buf[iq] >> (ir)) & 1) - ((buf[iq + N8] >> (ir)) & 1)) <<
//1 + Q;
//	}
//
//}

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

// void poly_sample_avx(uint16_t *r, unsigned char * seed, unsigned char nonce)
//
//{
//	int i, j, iq, ir;
//	unsigned char extseed[PKE_SEED_BYTES + 1], buf[N4];
//	for (i = 0; i < PKE_SEED_BYTES; i++)
//		extseed[i] = seed[i];
//	extseed[PKE_SEED_BYTES] = nonce;
//	shake256(buf, N4, extseed, PKE_SEED_BYTES + 1);
//
//	CBD_avx(r, buf);
//
//}

int test_poly() {
  LARGE_INTEGER start, end;
  LARGE_INTEGER frequency;

  uint16_t a[1024], b[1024], c[1024], xc[1024];
  unsigned char ta[1536], tb[1536], tc[1536];

  int i, j;
  double elapsed;

  QueryPerformanceFrequency(&frequency);

  double quadpart = (double)frequency.QuadPart;

  srand(time(NULL));

  for (i = 0; i < 10000; i++) {
    for (j = 0; j < 1024; j++) {
      a[j] = rand() % Q;
    }

    for (j = 0; j < 1024; j++) {
      b[j] = rand() % Q;
    }

    poly_add_avx(c, a, b);
    poly_add(xc, a, b);

    for (j = 0; j < 1024; j++) {
      if (c[j] != xc[j]) {
        return -4;
      }
    }
  }

  for (j = 0; j < 1024; j++) {
    a[j] = rand() & 0xfff;
  }
  QueryPerformanceCounter(&start);
  for (j = 0; j < 1000000; j++) {
    poly_tobyte(ta, a);
  }
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;
  printf("Poly2Bytes RUNS in Time : %0.6f usecs\n ", elapsed);

  QueryPerformanceCounter(&start);
  for (j = 0; j < 1000000; j++) {
    poly_tobyte_avx(ta, a);
  }
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;
  printf("  Poly2Bytes AVX RUNS in Time : %0.6f usecs\n\n\n", elapsed);

  for (j = 0; j < 1536; j++) {
    ta[j] = rand() & 0xff;
  }

  QueryPerformanceCounter(&start);

  for (j = 0; j < 1000000; j++) {
    byte_topoly(ta, a);
  }
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;

  printf("Byte2Poly RUNS in Time : %0.6f usecs\n ", elapsed);

  QueryPerformanceCounter(&start);
  for (j = 0; j < 1000000; j++) {
    byte_topoly_avx(ta, a);
  }
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;

  printf("   Byte2Poly AVX RUNS in Time : %0.6f usecs\n\n\n", elapsed);

#ifdef test_correctness

  for (i = 0; i < 100000; i++) {
    for (j = 0; j < 1024; j++) {
      a[j] = rand() & 0xfff;
      b[j] = a[j];
      c[j] = b[j];
    }
    poly_tobyte(ta, a);
    poly_tobyte_avx(tb, b);

    for (j = 0; j < 1536; j++) {
      if (ta[j] != tb[j]) {

        return -1;
      }
    }
    byte_topoly_avx(tb, c);
    for (j = 0; j < 1024; j++) {
      if (b[j] != c[j]) {
        printf("%d\t%02x\t%02x\n", j, ta[j], tb[j]);
        printf("%04x\t%04x\n", a[48], a[0]);
        return -3;
      }
    }

    for (j = 0; j < 1536; j++) {
      ta[j] = rand() & 0xff;
      tb[j] = ta[j];
    }
    byte_topoly(ta, a);
    byte_topoly_avx(tb, b);
    for (j = 0; j < 1024; j++) {
      if (a[j] != b[j]) {
        printf("%d\t%04x\t%04x\n", j, a[j], b[j]);
        printf("%d\t%04x\t%04x\n", j, a[j + 1], b[j + 1]);
        printf("%02x\t%02x\n", ta[48], ta[0]);
        return -2;
      }
    }
  }
#endif

  return 0;
}

/*************************************************
 * Name:        verify
 *
 * Description: Compare two arrays for equality in constant time.
 *
 * Arguments:   const unsigned char *a: pointer to first byte array
 *              const unsigned char *b: pointer to second byte array
 *              size_t len:             length of the byte arrays
 *
 * Returns 0 if the byte arrays are equal, 1 otherwise
 **************************************************/

int verify(const unsigned char *a, const unsigned char *b, size_t len) {
  int r;
  size_t i;
  r = 0;

  for (i = 0; i < len; i++)
    r |= a[i] ^ b[i];

  r = (-r) >> 31;
  return r;
}

int verify_avx(const unsigned char *a, const unsigned char *b, size_t len) {
  int r;
  size_t i;
  r = 0;
  __declspec(align(32)) unsigned char t[32];

  __m256i A, B, C = _mm256_setzero_si256();
  for (i = 0; i < (len >> 5); i++) {
    A = _mm256_loadu_si256((__m256i *)(a + 32 * i));
    B = _mm256_loadu_si256((__m256i *)(b + 32 * i));
    A = _mm256_xor_si256(A, B);
    C = _mm256_or_si256(A, C);
  }
  _mm256_store_si256((__m256i *)t, C);

  for (i = 0; i < 32; i += 2)
    r |= t[i] ^ t[i + 1];

  r = (-r) >> 31;
  return r;
}

/*************************************************
 * Name:        cmov
 *
 * Description: Copy len bytes from x to r if b is 1;
 *              don't modify x if b is 0. Requires b to be in {0,1};
 *              assumes two's complement representation of negative integers.
 *              Runs in constant time.
 *
 * Arguments:   unsigned char *r:       pointer to output byte array
 *              const unsigned char *x: pointer to input byte array
 *              size_t len:             Amount of bytes to be copied
 *              unsigned char b:        Condition bit; has to be in {0,1}
 **************************************************/

void cmov(unsigned char *r, const unsigned char *x, size_t len,
          unsigned char b) {
  size_t i;

  b = -b;
  for (i = 0; i < len; i++)
    r[i] ^= b & (x[i] ^ r[i]);
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
  printf("CBD_AVX RUNS in Time : %0.6f usecs\n", elapsed);

  QueryPerformanceCounter(&start);
  for (j = 0; j < 1000000; j++)
    CBD_ref(b, bytes);
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;
  printf("CBD_REF RUNS in Time : %0.6f usecs\n \n\n", elapsed);

  for (i = 0; i < 1024; i++) {
    if (a[i] % 3329 != b[i] % 3329) {
      return -1;
    }
  }

  return 0;
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
