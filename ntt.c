#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>

#include "ntt.h"
#include "parameter.h"

// max zetas[] = 3321,  max_zetas_inv[] = 3326;
#define qinv 3327
#define BCON 20159

#define test_time 1

// aligning allows faster memory access;
__declspec(align(32)) const uint16_t zetas[128] = {
    2285, 2571, 2970, 1812, 1493, 1422, 287,  202,  3158, 622,  1577, 182,
    962,  2127, 1855, 1468, 573,  2004, 264,  383,  2500, 1458, 1727, 3199,
    2648, 1017, 732,  608,  1787, 411,  3124, 1758, 1223, 652,  2777, 1015,
    2036, 1491, 3047, 1785, 516,  3321, 3009, 2663, 1711, 2167, 126,  1469,
    2476, 3239, 3058, 830,  107,  1908, 3082, 2378, 2931, 961,  1821, 2604,
    448,  2264, 677,  2054, 2226, 430,  555,  843,  2078, 871,  1550, 105,
    422,  587,  177,  3094, 3038, 2869, 1574, 1653, 3083, 778,  1159, 3182,
    2552, 1483, 2727, 1119, 1739, 644,  2457, 349,  418,  329,  3173, 3254,
    817,  1097, 603,  610,  1322, 2044, 1864, 384,  2114, 3193, 1218, 1994,
    2455, 220,  2142, 1670, 2144, 1799, 2051, 794,  1819, 2475, 2459, 478,
    3221, 3021, 996,  991,  958,  1869, 1522, 1628};
__declspec(align(32)) const uint16_t zetas_inv[128] = {
    512,  2380, 140,  1379, 2431, 143,  1575, 3030, 1549, 2441, 1906, 2462,
    3278, 3326, 3133, 2730, 1923, 2463, 1124, 2416, 2492, 1909, 2658, 548,
    1403, 670,  1606, 2836, 3300, 2544, 3087, 1944, 506,  1988, 2271, 1896,
    699,  2391, 3078, 1356, 2038, 1099, 3002, 1939, 1289, 3209, 2147, 2672,
    353,  1979, 1683, 99,   3139, 3122, 3121, 1946, 2856, 168,  989,  254,
    2169, 1890, 307,  1193, 266,  2953, 957,  2602, 1328, 2428, 3276, 976,
    1624, 683,  236,  993,  1625, 1858, 2655, 352,  804,  2593, 740,  631,
    2387, 1707, 1667, 1273, 1054, 62,   2941, 173,  206,  2362, 2293, 1114,
    653,  1605, 1661, 881,  3185, 579,  1209, 2421, 1709, 688,  2782, 3101,
    1749, 1082, 3001, 764,  2199, 521,  2968, 1937, 2268, 1700, 100,  985,
    2212, 2480, 1125, 262,  1582, 1268, 1837, 1283};
__declspec(align(32)) const uint16_t omegas_inv_bitrev[64] = {
    2285, 758,  1517, 359,  3127, 3042, 1907, 1836, 1861, 1474, 1202,
    2367, 3147, 1752, 2707, 171,  1571, 205,  2918, 1542, 2721, 2597,
    2312, 681,  130,  1602, 1871, 829,  2946, 3065, 1325, 2756, 1275,
    2652, 1065, 2881, 725,  1508, 2368, 398,  951,  247,  1421, 3222,
    2499, 271,  90,   853,  1860, 3203, 1162, 1618, 666,  320,  8,
    2813, 1544, 282,  1838, 1293, 2314, 552,  2677, 2106};

//__declspec(align(32)) const uint16_t xQ[16] = {
//Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q };
//__declspec(align(32)) const uint16_t xBCON[16] = { BCON, BCON, BCON, BCON,
//BCON, BCON, BCON, BCON, BCON, BCON, BCON, BCON, BCON, BCON, BCON, BCON };

__declspec(align(32)) const uint16_t zetas_avx[256] = {
    3158, 3158, 3158, 3158, 3158, 3158, 3158, 3158, 622,  622,  622,  622,
    622,  622,  622,  622,  1577, 1577, 1577, 1577, 1577, 1577, 1577, 1577,
    182,  182,  182,  182,  182,  182,  182,  182,  962,  962,  962,  962,
    962,  962,  962,  962,  2127, 2127, 2127, 2127, 2127, 2127, 2127, 2127,
    1855, 1855, 1855, 1855, 1855, 1855, 1855, 1855, 1468, 1468, 1468, 1468,
    1468, 1468, 1468, 1468, 264,  264,  264,  264,  573,  573,  573,  573,
    383,  383,  383,  383,  2004, 2004, 2004, 2004, 1727, 1727, 1727, 1727,
    2500, 2500, 2500, 2500, 3199, 3199, 3199, 3199, 1458, 1458, 1458, 1458,
    732,  732,  732,  732,  2648, 2648, 2648, 2648, 608,  608,  608,  608,
    1017, 1017, 1017, 1017, 3124, 3124, 3124, 3124, 1787, 1787, 1787, 1787,
    1758, 1758, 1758, 1758, 411,  411,  411,  411,  2036, 2036, 1223, 1223,
    1491, 1491, 652,  652,  3047, 3047, 2777, 2777, 1785, 1785, 1015, 1015,
    1711, 1711, 516,  516,  2167, 2167, 3321, 3321, 126,  126,  3009, 3009,
    1469, 1469, 2663, 2663, 107,  107,  2476, 2476, 1908, 1908, 3239, 3239,
    3082, 3082, 3058, 3058, 2378, 2378, 830,  830,  448,  448,  2931, 2931,
    2264, 2264, 961,  961,  677,  677,  1821, 1821, 2054, 2054, 2604, 2604,
    422,  2226, 587,  430,  177,  555,  3094, 843,  3038, 2078, 2869, 871,
    1574, 1550, 1653, 105,  1739, 3083, 644,  778,  2457, 1159, 349,  3182,
    418,  2552, 329,  1483, 3173, 2727, 3254, 1119, 2114, 817,  3193, 1097,
    1218, 603,  1994, 610,  2455, 1322, 220,  2044, 2142, 1864, 1670, 384,
    3221, 2144, 3021, 1799, 996,  2051, 991,  794,  958,  1819, 1869, 2475,
    1522, 2459, 1628, 478};

__declspec(align(32)) const uint16_t omegas_avx[256] = {
    1861, 2285, 1474, 758,  1202, 1517, 2367, 359,  3147, 3127, 1752, 3042,
    2707, 1907, 171,  1836, 130,  1571, 1602, 205,  1871, 2918, 829,  1542,
    2946, 2721, 3065, 2597, 1325, 2312, 2756, 681,  951,  1275, 247,  2652,
    1421, 1065, 3222, 2881, 2499, 725,  271,  1508, 90,   2368, 853,  398,
    1544, 1860, 282,  3203, 1838, 1162, 1293, 1618, 2314, 666,  552,  320,
    2677, 8,    2106, 2813, 3127, 3127, 2285, 2285, 3042, 3042, 758,  758,
    1907, 1907, 1517, 1517, 1836, 1836, 359,  359,  3147, 3147, 1861, 1861,
    1752, 1752, 1474, 1474, 2707, 2707, 1202, 1202, 171,  171,  2367, 2367,
    2721, 2721, 1571, 1571, 2597, 2597, 205,  205,  2312, 2312, 2918, 2918,
    681,  681,  1542, 1542, 2946, 2946, 130,  130,  3065, 3065, 1602, 1602,
    1325, 1325, 1871, 1871, 2756, 2756, 829,  829,  1517, 1517, 1517, 1517,
    2285, 2285, 2285, 2285, 359,  359,  359,  359,  758,  758,  758,  758,
    1907, 1907, 1907, 1907, 3127, 3127, 3127, 3127, 1836, 1836, 1836, 1836,
    3042, 3042, 3042, 3042, 1202, 1202, 1202, 1202, 1861, 1861, 1861, 1861,
    2367, 2367, 2367, 2367, 1474, 1474, 1474, 1474, 2707, 2707, 2707, 2707,
    3147, 3147, 3147, 3147, 171,  171,  171,  171,  1752, 1752, 1752, 1752,
    2285, 2285, 2285, 2285, 2285, 2285, 2285, 2285, 758,  758,  758,  758,
    758,  758,  758,  758,  1517, 1517, 1517, 1517, 1517, 1517, 1517, 1517,
    359,  359,  359,  359,  359,  359,  359,  359,  3127, 3127, 3127, 3127,
    3127, 3127, 3127, 3127, 3042, 3042, 3042, 3042, 3042, 3042, 3042, 3042,
    1907, 1907, 1907, 1907, 1907, 1907, 1907, 1907, 1836, 1836, 1836, 1836,
    1836, 1836, 1836, 1836};

__declspec(align(32)) const uint16_t ntt_y_avx[] = {
    17,   3312, 2761, 568,  583,  2746, 2649, 680,  1637, 1692, 723,  2606,
    2288, 1041, 1100, 2229, 1409, 1920, 2662, 667,  3281, 48,   233,  3096,
    756,  2573, 2156, 1173, 3015, 314,  3050, 279,  1703, 1626, 1651, 1678,
    2789, 540,  1789, 1540, 1847, 1482, 952,  2377, 1461, 1868, 2687, 642,
    939,  2390, 2308, 1021, 2437, 892,  2388, 941,  733,  2596, 2337, 992,
    268,  3061, 641,  2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109,
    375,  2954, 2549, 780,  2090, 1239, 1645, 1684, 1063, 2266, 319,  3010,
    2773, 556,  757,  2572, 2099, 1230, 561,  2768, 2466, 863,  2594, 735,
    2804, 525,  1092, 2237, 403,  2926, 1026, 2303, 1143, 2186, 2150, 1179,
    2775, 554,  886,  2443, 1722, 1607, 1212, 2117, 1874, 1455, 1029, 2300,
    2110, 1219, 2935, 394,  885,  2444, 2154, 1175};

static const uint32_t rlog = 16;

// Compute the Mont Reduce of yin1 * yin2
#define MReduce(yout, yin1, yin2, y9, ya, yb, yc, yd, ye, yf)                  \
  {                                                                            \
    y9 = _mm256_mulhi_epu16(yin1, yin2);                                       \
    ya = _mm256_mullo_epi16(yin1, yin2);                                       \
    yb = _mm256_mullo_epi16(qinv_avx, ya);                                     \
    yc = _mm256_mulhi_epu16(yb, xQ_avx);                                       \
    yd = _mm256_mullo_epi16(yb, xQ_avx);                                       \
    y9 = _mm256_add_epi16(y9, yc);                                             \
    ye = _mm256_xor_si256(ya, yd);                                             \
    yf = _mm256_adds_epu16(ya, yd);                                            \
    ya = _mm256_cmpeq_epi16(ye, mask_all);                                     \
    yb = _mm256_cmpeq_epi16(yf, mask_all);                                     \
    yc = _mm256_andnot_si256(ya, yb);                                          \
    yout = _mm256_sub_epi16(y9, yc);                                           \
  }

#define BReduce(yout, yin, ya)                                                 \
  {                                                                            \
    ya = _mm256_mulhi_epu16(BC_avx, yin);                                      \
    ya = _mm256_srli_epi16(ya, 10);                                            \
    ya = _mm256_mullo_epi16(ya, xQ_avx);                                       \
    yout = _mm256_sub_epi16(yin, ya);                                          \
  }

#define AddMul(yout, xin1, xin2, yin1, yin2, th, tl, temp2, temp3, temp0,      \
               temp1)                                                          \
  {                                                                            \
    temp2 = _mm256_add_epi16(xin1, xin2);                                      \
    temp3 = _mm256_add_epi16(yin1, yin2);                                      \
    th = _mm256_mulhi_epu16(temp2, temp3);                                     \
    tl = _mm256_mullo_epi16(temp2, temp3);                                     \
    temp0 = _mm256_mulhi_epu16(BC_avx, tl);                                    \
    temp0 = _mm256_srli_epi16(temp0, 10);                                      \
    temp0 = _mm256_mullo_epi16(temp0, xQ_avx);                                 \
    tl = _mm256_sub_epi16(tl, temp0);                                          \
    temp0 = _mm256_mullo_epi16(MC_avx, th);                                    \
    temp0 = _mm256_add_epi16(Q4_avx, temp0);                                   \
    temp1 = _mm256_mullo_epi16(_mm256_set1_epi16(11), th);                     \
    temp1 = _mm256_srli_epi16(temp1, 4);                                       \
    temp1 = _mm256_mullo_epi16(xQ_avx, temp1);                                 \
    th = _mm256_sub_epi16(temp0, temp1);                                       \
    yout = _mm256_add_epi16(tl, th);                                           \
  }

#define MulY(yout, yin, th, tl, temp0, temp1)                                  \
  {                                                                            \
    th = _mm256_mulhi_epu16(yin, Y);                                           \
    tl = _mm256_mullo_epi16(yin, Y);                                           \
    temp0 = _mm256_mulhi_epu16(BC_avx, tl);                                    \
    temp0 = _mm256_srli_epi16(temp0, 10);                                      \
    temp0 = _mm256_mullo_epi16(temp0, xQ_avx);                                 \
    tl = _mm256_sub_epi16(tl, temp0);                                          \
    temp0 = _mm256_mullo_epi16(MC_avx, th);                                    \
    temp0 = _mm256_add_epi16(Q4_avx, temp0);                                   \
    temp1 = _mm256_mullo_epi16(_mm256_set1_epi16(11), th);                     \
    temp1 = _mm256_srli_epi16(temp1, 4);                                       \
    temp1 = _mm256_mullo_epi16(xQ_avx, temp1);                                 \
    th = _mm256_sub_epi16(temp0, temp1);                                       \
    yout = _mm256_add_epi16(tl, th);                                           \
  }

// when a < 2^28;  Output < 7424 < 4Q
// when a < 2^27;  Output < 5376 < 2Q
// when a < 2^26;  Output < 4352 < 2Q
uint16_t montgomery_reduce(uint32_t a) {
  uint32_t u;

  // Here u only compute the lower 16 bits of a * qinv
  u = (a * qinv);
  u &= ((1 << rlog) - 1);

  u *= Q;
  a = (a + u) >> rlog;
  return a;
}

uint16_t barrett_reduce(uint16_t a) {
  int32_t t;
  t = BCON * a;
  t >>= 26;
  t *= Q;
  return a - t;
}

void poly_pointwise(uint16_t *h, uint16_t *f, uint16_t *g) {

  int i, j;
  uint16_t ntt_y[N8] = {
      17,   3312, 2761, 568,  583,  2746, 2649, 680,  1637, 1692, 723,  2606,
      2288, 1041, 1100, 2229, 1409, 1920, 2662, 667,  3281, 48,   233,  3096,
      756,  2573, 2156, 1173, 3015, 314,  3050, 279,  1703, 1626, 1651, 1678,
      2789, 540,  1789, 1540, 1847, 1482, 952,  2377, 1461, 1868, 2687, 642,
      939,  2390, 2308, 1021, 2437, 892,  2388, 941,  733,  2596, 2337, 992,
      268,  3061, 641,  2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109,
      375,  2954, 2549, 780,  2090, 1239, 1645, 1684, 1063, 2266, 319,  3010,
      2773, 556,  757,  2572, 2099, 1230, 561,  2768, 2466, 863,  2594, 735,
      2804, 525,  1092, 2237, 403,  2926, 1026, 2303, 1143, 2186, 2150, 1179,
      2775, 554,  886,  2443, 1722, 1607, 1212, 2117, 1874, 1455, 1029, 2300,
      2110, 1219, 2935, 394,  885,  2444, 2154, 1175};
  uint16_t tf[8], tg[8];
  uint16_t y;
  uint32_t tfg[8];

  for (i = 0; i < N8; i++) {
    tf[0] = f[i];
    tf[1] = f[128 + i];
    tf[2] = f[256 + i];
    tf[3] = f[384 + i];
    tf[4] = f[512 + i];
    tf[5] = f[640 + i];
    tf[6] = f[768 + i];
    tf[7] = f[896 + i];
    tg[0] = g[i];
    tg[1] = g[128 + i];
    tg[2] = g[256 + i];
    tg[3] = g[384 + i];
    tg[4] = g[512 + i];
    tg[5] = g[640 + i];
    tg[6] = g[768 + i];
    tg[7] = g[896 + i];

    tfg[0] = tf[0] * tg[0];
    tfg[1] = tf[1] * tg[1];
    tfg[2] = tf[2] * tg[2];
    tfg[3] = tf[3] * tg[3];
    tfg[4] = tf[4] * tg[4];
    tfg[5] = tf[5] * tg[5];
    tfg[6] = tf[6] * tg[6];
    tfg[7] = tf[7] * tg[7];

    y = ntt_y[i];
    h[i] =
        (tfg[0] + (y * (((tf[1] + tf[7]) * (tg[1] + tg[7]) +
                         (tf[2] + tf[6]) * (tg[2] + tg[6]) +
                         (tf[3] + tf[5]) * (tg[3] + tg[5]) - tfg[1] - tfg[7] +
                         tfg[4] - tfg[2] - tfg[6] - tfg[3] - tfg[5]) %
                        Q))) %
        Q;
    h[128 + i] = (((tf[0] + tf[1]) * (tg[0] + tg[1]) - tfg[0] - tfg[1]) +
                  y * (((tf[2] + tf[7]) * (tg[2] + tg[7]) - tfg[2] - tfg[7] +
                        (tf[3] + tf[6]) * (tg[3] + tg[6]) - tfg[3] - tfg[6] +
                        (tf[4] + tf[5]) * (tg[4] + tg[5]) - tfg[4] - tfg[5]) %
                       Q)) %
                 Q;
    h[256 + i] =
        (tfg[1] + ((tf[0] + tf[2]) * (tg[0] + tg[2]) - tfg[0] - tfg[2]) +
         y * ((tfg[5] + (tf[3] + tf[7]) * (tg[3] + tg[7]) - tfg[3] - tfg[7] +
               (tf[4] + tf[6]) * (tg[4] + tg[6]) - tfg[4] - tfg[6]) %
              Q)) %
        Q;
    h[384 + i] = (((tf[0] + tf[3]) * (tg[0] + tg[3]) - tfg[0] - tfg[3]) +
                  ((tf[1] + tf[2]) * (tg[1] + tg[2]) - tfg[1] - tfg[2]) +
                  y * (((tf[4] + tf[7]) * (tg[4] + tg[7]) - tfg[4] - tfg[7] +
                        (tf[5] + tf[6]) * (tg[5] + tg[6]) - tfg[5] - tfg[6]) %
                       Q)) %
                 Q;
    h[512 + i] =
        (tfg[2] + ((tf[0] + tf[4]) * (tg[0] + tg[4]) - tfg[0] - tfg[4]) +
         ((tf[1] + tf[3]) * (tg[1] + tg[3]) - tfg[1] - tfg[3]) +
         y * ((tfg[6] + (tf[5] + tf[7]) * (tg[5] + tg[7]) - tfg[5] - tfg[7]) %
              Q)) %
        Q;
    h[640 + i] =
        (((tf[0] + tf[5]) * (tg[0] + tg[5]) - tfg[0] - tfg[5]) +
         ((tf[1] + tf[4]) * (tg[1] + tg[4]) - tfg[1] - tfg[4]) +
         ((tf[2] + tf[3]) * (tg[2] + tg[3]) - tfg[2] - tfg[3]) +
         y * (((tf[6] + tf[7]) * (tg[6] + tg[7]) - tfg[6] - tfg[7]) % Q)) %
        Q;
    h[768 + i] =
        (tfg[3] + ((tf[0] + tf[6]) * (tg[0] + tg[6]) - tfg[0] - tfg[6]) +
         ((tf[1] + tf[5]) * (tg[1] + tg[5]) - tfg[1] - tfg[5]) +
         ((tf[2] + tf[4]) * (tg[2] + tg[4]) - tfg[2] - tfg[4]) +
         y * (tfg[7] % Q)) %
        Q;
    h[896 + i] = (((tf[0] + tf[7]) * (tg[0] + tg[7]) - tfg[0] - tfg[7]) +
                  ((tf[1] + tf[6]) * (tg[1] + tg[6]) - tfg[1] - tfg[6]) +
                  ((tf[2] + tf[5]) * (tg[2] + tg[5]) - tfg[2] - tfg[5]) +
                  ((tf[3] + tf[4]) * (tg[3] + tg[4]) - tfg[3] - tfg[4])) %
                 Q;
  }
}

void poly_pointwise_avx(uint16_t *h, uint16_t *f, uint16_t *g) {
  int i, j;

  __m256i xQ_avx = _mm256_set1_epi16(Q);
  __m256i mask_all = _mm256_set1_epi16(0xffff);
  __m256i qinv_avx = _mm256_set1_epi16(qinv);
  __m256i Q2_avx = _mm256_set1_epi16(Q * 2);
  __m256i Q4_avx = _mm256_set1_epi16(Q * 4);
  __m256i Q6_avx = _mm256_set1_epi16(Q * 6);
  __m256i Q8_avx = _mm256_set1_epi16(Q * 8);
  __m256i BC_avx = _mm256_set1_epi16(BCON);
  __m256i MC_avx = _mm256_set1_epi16(2285);

  __m256i y[8], x[8], th[8], tl[8], temp[8], prod[8], Y; // hi[8], lo[8],
  __m256i z[8];
  for (i = 0; i < 8; i++) {
    // compute tfg[i] = f[i] * g[i]%Q
    for (j = 0; j < 8; j++) {
      x[j] = _mm256_loadu_si256((__m256i *)(f + 128 * j + 16 * i));
      y[j] = _mm256_loadu_si256((__m256i *)(g + 128 * j + 16 * i));

      th[j] = _mm256_mulhi_epu16(x[j], y[j]);
      tl[j] = _mm256_mullo_epi16(x[j], y[j]);

      // Reduce (th, tl) = x * y; That is tfg[] in the original implementation
      // Barrett Reduce the lower 16 bits;
      BReduce(prod[j], tl[j], temp[0]);

      // Montgomery-like reduce the higher 16 bits;
      temp[0] = _mm256_mullo_epi16(MC_avx, th[j]);
      temp[0] = _mm256_add_epi16(Q4_avx, temp[0]);
      temp[1] = _mm256_mullo_epi16(_mm256_set1_epi16(11), th[j]);
      temp[1] = _mm256_srli_epi16(temp[1], 4);
      temp[1] = _mm256_mullo_epi16(xQ_avx, temp[1]);
      temp[1] = _mm256_sub_epi16(temp[0], temp[1]);
      // Combine the result; The max possible of prod now is 19717 < 6Q;
      prod[j] = _mm256_add_epi16(prod[j], temp[1]);

      // Barrett Reduce the product;
      BReduce(prod[j], prod[j], temp[0]);
    }

    // h[i] =
    //	(
    //		tfg[0] +
    //		(
    //			y *
    //			(
    //			(
    //				(tf[1] + tf[7]) * (tg[1] + tg[7])
    //				+ (tf[2] + tf[6])*(tg[2] + tg[6])
    //				+ (tf[3] + tf[5])*(tg[3] + tg[5])
    //				- tfg[1] - tfg[7] + tfg[4] - tfg[2] - tfg[6] - tfg[3] -
    //tfg[5] 				) % Q
    //				)
    //			)
    //		) % Q;

    Y = _mm256_load_si256((__m256i *)(ntt_y_avx + 16 * i));
    AddMul(temp[4], x[1], x[7], y[1], y[7], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    AddMul(temp[5], x[2], x[6], y[2], y[6], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    // Now temp[4] < 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    AddMul(temp[5], x[3], x[5], y[3], y[5], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    // Now temp[4] < 18Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);
    temp[4] = _mm256_add_epi16(temp[4], prod[4]);

    BReduce(temp[4], temp[4], temp[0]);

    temp[0] = _mm256_add_epi16(prod[1], prod[7]);
    temp[1] = _mm256_add_epi16(prod[3], prod[5]);
    temp[2] = _mm256_add_epi16(prod[2], prod[6]);
    temp[3] = _mm256_add_epi16(temp[0], temp[1]);
    temp[0] = _mm256_add_epi16(temp[2], temp[3]);

    temp[4] = _mm256_add_epi16(temp[4], Q6_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    // Now brought temp[4] down to Q
    BReduce(temp[4], temp[4], temp[0]);

    // Now comput temp[4] * Y mod Q
    MulY(temp[4], temp[4], th[0], tl[0], temp[0], temp[1]);

    temp[4] = _mm256_add_epi16(temp[4], prod[0]);

    BReduce(temp[4], temp[4], temp[0]);
    _mm256_storeu_si256((__m256i *)(h + 16 * i), temp[4]);

    // Now the second H[]
    // h[128 + i] =
    //	(
    //	(tf[0] + tf[1]) * (tg[0] + tg[1]) - tfg[0] - tfg[1]
    //		+ y *
    //		(
    //		(
    //			(tf[2] + tf[7]) * (tg[2] + tg[7]) - tfg[2] - tfg[7]
    //			+ (tf[3] + tf[6])*(tg[3] + tg[6]) - tfg[3] - tfg[6]
    //			+ (tf[4] + tf[5])*(tg[4] + tg[5]) - tfg[4] - tfg[5]
    //			) % Q
    //			)
    //		) % Q;

    AddMul(temp[4], x[2], x[7], y[2], y[7], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    AddMul(temp[5], x[3], x[6], y[3], y[6], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    // Now temp[4] < 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    AddMul(temp[5], x[4], x[5], y[4], y[5], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    // Now temp[4] < 18Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    BReduce(temp[4], temp[4], temp[0]);

    temp[0] = _mm256_add_epi16(prod[2], prod[7]);
    temp[1] = _mm256_add_epi16(prod[3], prod[6]);
    temp[2] = _mm256_add_epi16(prod[4], prod[5]);
    temp[3] = _mm256_add_epi16(temp[0], temp[1]);
    temp[0] = _mm256_add_epi16(temp[2], temp[3]);

    temp[0] = _mm256_sub_epi16(Q6_avx, temp[0]);

    // 7Q
    temp[4] = _mm256_add_epi16(temp[4], temp[0]);

    BReduce(temp[4], temp[4], temp[0]);

    MulY(temp[4], temp[4], th[0], tl[0], temp[0], temp[1]);

    AddMul(temp[5], x[0], x[1], y[0], y[1], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    // 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);
    temp[0] = _mm256_add_epi16(prod[0], prod[1]);
    temp[4] = _mm256_add_epi16(temp[4], Q2_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    BReduce(temp[4], temp[4], temp[0]);
    _mm256_storeu_si256((__m256i *)(h + 16 * i + 128), temp[4]);

    // Now the Third H[]
    // h[256 + i] =
    //	(
    //		tfg[1]
    //		+ (tf[0] + tf[2])*(tg[0] + tg[2]) - tfg[0] - tfg[2]
    //		+ y *
    //		(
    //		(
    //			tfg[5] + (tf[3] + tf[7]) * (tg[3] + tg[7]) - tfg[3] -
    //tfg[7]
    //			+ (tf[4] + tf[6])*(tg[4] + tg[6]) - tfg[4] - tfg[6]
    //			) % Q
    //			)
    //		) % Q;

    AddMul(temp[4], x[3], x[7], y[3], y[7], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    AddMul(temp[5], x[4], x[6], y[4], y[6], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    // 13Q
    temp[4] = _mm256_add_epi16(temp[4], prod[5]);

    temp[0] = _mm256_add_epi16(prod[3], prod[7]);
    temp[1] = _mm256_add_epi16(prod[4], prod[6]);
    temp[0] = _mm256_add_epi16(temp[0], temp[1]);
    // 17Q
    temp[4] = _mm256_add_epi16(temp[4], Q4_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);
    BReduce(temp[4], temp[4], temp[0]);

    MulY(temp[4], temp[4], th[0], tl[0], temp[0], temp[1]);
    AddMul(temp[0], x[0], x[2], y[0], y[2], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[0]);

    temp[0] = _mm256_add_epi16(prod[0], prod[2]);
    temp[4] = _mm256_add_epi16(temp[4], Q2_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    temp[4] = _mm256_add_epi16(temp[4], prod[1]);

    BReduce(temp[4], temp[4], temp[0]);
    _mm256_storeu_si256((__m256i *)(h + 16 * i + 256), temp[4]);

    // Now the Fourth H[]
    // h[384 + i] =
    //	(
    //	(
    //		(tf[0] + tf[3])*(tg[0] + tg[3]) - tfg[0] - tfg[3]
    //		+ ((tf[1] + tf[2])*(tg[1] + tg[2]) - tfg[1] - tfg[2]
    //			+ y *
    //			(

    //			(tf[4] + tf[7])*(tg[4] + tg[7]) - tfg[4] - tfg[7] + (tf[5] +
    //tf[6])*(tg[5] + tg[6]) - tfg[5] - tfg[6] 				) % Q
    //			)
    //		) % Q;

    AddMul(temp[4], x[4], x[7], y[4], y[7], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    AddMul(temp[5], x[5], x[6], y[5], y[6], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    temp[0] = _mm256_add_epi16(prod[4], prod[7]);
    temp[1] = _mm256_add_epi16(prod[5], prod[6]);
    temp[0] = _mm256_add_epi16(temp[0], temp[1]);
    // 16Q
    temp[4] = _mm256_add_epi16(temp[4], Q4_avx);

    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    BReduce(temp[4], temp[4], temp[0]);

    MulY(temp[4], temp[4], th[0], tl[0], temp[0], temp[1]);

    AddMul(temp[5], x[0], x[3], y[0], y[3], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    AddMul(temp[5], x[1], x[2], y[1], y[2], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // 18Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    BReduce(temp[4], temp[4], temp[0]);

    temp[0] = _mm256_add_epi16(prod[0], prod[3]);
    temp[1] = _mm256_add_epi16(prod[1], prod[2]);
    temp[0] = _mm256_add_epi16(temp[1], temp[0]);

    temp[4] = _mm256_add_epi16(temp[4], Q4_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    BReduce(temp[4], temp[4], temp[0]);
    _mm256_storeu_si256((__m256i *)(h + 16 * i + 384), temp[4]);

    // Now the Fifth H[]
    // h[512 + i] =
    //	(
    //		tfg[2] + (tf[0] + tf[4])*(tg[0] + tg[4]) - tfg[0] - tfg[4]
    //		+ (tf[1] + tf[3])*(tg[1] + tg[3]) - tfg[1] - tfg[3]
    //		+ y *
    //		(
    //			tfg[6] + (tf[5] + tf[7])*(tg[5] + tg[7]) - tfg[5] -
    //tfg[7] 			) % Q 		) % Q;

    AddMul(temp[4], x[5], x[7], y[5], y[7], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    temp[0] = _mm256_add_epi16(prod[5], prod[7]);
    temp[4] = _mm256_add_epi16(temp[4], Q2_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);
    temp[4] = _mm256_add_epi16(temp[4], prod[6]);

    BReduce(temp[4], temp[4], temp[0]);
    MulY(temp[4], temp[4], th[0], tl[0], temp[0], temp[1]);

    AddMul(temp[5], x[0], x[4], y[0], y[4], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    AddMul(temp[5], x[1], x[3], y[1], y[3], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 18Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    BReduce(temp[4], temp[4], temp[0]);

    temp[0] = _mm256_add_epi16(prod[0], prod[4]);
    temp[1] = _mm256_add_epi16(prod[1], prod[3]);
    temp[0] = _mm256_add_epi16(temp[1], temp[0]);

    temp[4] = _mm256_add_epi16(temp[4], Q4_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);
    temp[4] = _mm256_add_epi16(temp[4], prod[2]);

    // Now brought temp[4] down to Q
    BReduce(temp[4], temp[4], temp[0]);

    _mm256_storeu_si256((__m256i *)(h + 16 * i + 512), temp[4]);

    // Now the Sixth H[]
    // h[640 + i] =
    //	(
    //	(tf[0] + tf[5])*(tg[0] + tg[5]) - tfg[0] - tfg[5]
    //		+ (tf[1] + tf[4])*(tg[1] + tg[4]) - tfg[1] - tfg[4]
    //		+ (tf[2] + tf[3])*(tg[2] + tg[3]) - tfg[2] - tfg[3]
    //		+ y *
    //		(
    //		(tf[6] + tf[7])*(tg[6] + tg[7]) - tfg[6] - tfg[7]
    //			) % Q

    //		) % Q;
    AddMul(temp[4], x[6], x[7], y[6], y[7], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    temp[0] = _mm256_add_epi16(prod[6], prod[7]);
    temp[4] = _mm256_add_epi16(temp[4], Q2_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    // Now brought temp[4] down to Q
    BReduce(temp[4], temp[4], temp[0]);

    // Now comput temp[4] * Y mod Q
    MulY(temp[4], temp[4], th[0], tl[0], temp[0], temp[1]);

    AddMul(temp[5], x[0], x[5], y[0], y[5], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    AddMul(temp[5], x[1], x[4], y[1], y[4], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 18Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    BReduce(temp[4], temp[4], temp[0]);

    AddMul(temp[5], x[2], x[3], y[2], y[3], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 7Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    temp[0] = _mm256_add_epi16(prod[0], prod[5]);
    temp[1] = _mm256_add_epi16(prod[1], prod[4]);
    temp[2] = _mm256_add_epi16(prod[2], prod[3]);
    temp[0] = _mm256_add_epi16(temp[0], temp[1]);
    temp[0] = _mm256_add_epi16(temp[0], temp[2]);

    temp[4] = _mm256_add_epi16(temp[4], Q6_avx);
    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    BReduce(temp[4], temp[4], temp[0]);
    _mm256_storeu_si256((__m256i *)(h + 16 * i + 640), temp[4]);

    // The Seventh H[]
    // h[768 + i] =
    //	(
    //		tfg[3]
    //		+ (tf[0] + tf[6])*(tg[0] + tg[6]) - tfg[0] - tfg[6] +
    //		(tf[1] + tf[5])*(tg[1] + tg[5]) - tfg[1] - tfg[5] +
    //		(tf[2] + tf[4])*(tg[2] + tg[4]) - tfg[2] - tfg[4] +
    //		y * tfg[7] % Q
    //		) % Q;

    // Now compute temp[4] * Y mod Q
    MulY(temp[4], prod[7], th[0], tl[0], temp[0], temp[1]);

    AddMul(temp[5], x[0], x[6], y[0], y[6], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);

    // 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    AddMul(temp[5], x[1], x[5], y[1], y[5], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 18Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    BReduce(temp[4], temp[4], temp[0]);

    AddMul(temp[5], x[2], x[4], y[2], y[4], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 7Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    temp[0] = _mm256_add_epi16(prod[0], prod[6]);
    temp[1] = _mm256_add_epi16(prod[1], prod[5]);
    temp[2] = _mm256_add_epi16(prod[2], prod[4]);
    temp[0] = _mm256_add_epi16(temp[0], temp[1]);
    temp[0] = _mm256_add_epi16(temp[0], temp[2]);

    temp[4] = _mm256_add_epi16(temp[4], Q6_avx);

    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);
    temp[4] = _mm256_add_epi16(temp[4], prod[3]);

    BReduce(temp[4], temp[4], temp[0]);
    _mm256_storeu_si256((__m256i *)(h + 16 * i + 768), temp[4]);

    // The last H[]
    // h[896 + i] =
    //	(
    //	(tf[0] + tf[7])*(tg[0] + tg[7]) - tfg[0] - tfg[7] +
    //		(tf[1] + tf[6])*(tg[1] + tg[6]) - tfg[1] - tfg[6] +
    //		(tf[2] + tf[5])*(tg[2] + tg[5]) - tfg[2] - tfg[5] +
    //		(tf[3] + tf[4])*(tg[3] + tg[4]) - tfg[3] - tfg[4]
    //		) % Q;

    AddMul(temp[4], x[0], x[7], y[0], y[7], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    AddMul(temp[5], x[1], x[6], y[1], y[6], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 12Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    AddMul(temp[5], x[2], x[5], y[2], y[5], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 18Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    BReduce(temp[4], temp[4], temp[0]);

    AddMul(temp[5], x[3], x[4], y[3], y[4], th[0], tl[0], temp[2], temp[3],
           temp[0], temp[1]);
    // Now temp[4] < 7Q
    temp[4] = _mm256_add_epi16(temp[4], temp[5]);

    temp[0] = _mm256_add_epi16(prod[0], prod[7]);
    temp[1] = _mm256_add_epi16(prod[1], prod[6]);
    temp[2] = _mm256_add_epi16(prod[2], prod[5]);
    temp[0] = _mm256_add_epi16(temp[0], temp[1]);
    temp[1] = _mm256_add_epi16(prod[3], prod[4]);
    temp[0] = _mm256_add_epi16(temp[0], temp[2]);
    temp[0] = _mm256_add_epi16(temp[0], temp[1]);

    temp[4] = _mm256_add_epi16(temp[4], Q8_avx);

    temp[4] = _mm256_sub_epi16(temp[4], temp[0]);

    BReduce(temp[4], temp[4], temp[0]);

    _mm256_storeu_si256((__m256i *)(h + 16 * i + 896), temp[4]);
  }
}

void poly_invntt_avx(uint16_t *p) {
  int start, j, jTwiddle, level;
  uint16_t temp, W, ttt[128];
  uint32_t t;

  __m256i xQ_avx = _mm256_set1_epi16(Q);
  __m256i mask_all = _mm256_set1_epi16(0xffff);
  __m256i qinv_avx = _mm256_set1_epi16(qinv);
  __m256i Q2_avx = _mm256_set1_epi16(Q * 2);
  __m256i Q4_avx = _mm256_set1_epi16(Q * 4);
  __m256i BC_avx = _mm256_set1_epi16(BCON);

  __m256i y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf;

  __m256i temp0, temp1, temp2, temp3;

  // Level 0
  y0 = _mm256_loadu_si256((__m256i *)p);
  y1 = _mm256_loadu_si256((__m256i *)(p + 16));

  __m256i mask_hi = _mm256_set1_epi32(0xffff0000);
  __m256i mask_lo = _mm256_set1_epi32(0xffff);

  temp0 = _mm256_slli_epi32(y0, 16);
  temp1 = _mm256_and_si256(y1, mask_lo);
  temp2 = _mm256_srli_epi32(y1, 16);
  temp3 = _mm256_and_si256(y0, mask_hi);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_add_epi16(y0, xQ_avx);
  temp1 = _mm256_sub_epi16(temp0, y1);
  y0 = _mm256_add_epi16(y0, y1);

  // Montgomery Reduce y8 * temp1
  y8 = _mm256_load_si256((__m256i *)omegas_avx);
  MReduce(y1, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_srli_epi32(y0, 16);
  temp1 = _mm256_and_si256(y1, mask_hi);
  temp2 = _mm256_slli_epi32(y1, 16);
  temp3 = _mm256_and_si256(y0, mask_lo);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  //(y2, y3)
  y3 = _mm256_loadu_si256((__m256i *)(p + 48));
  y2 = _mm256_loadu_si256((__m256i *)(p + 32));

  temp0 = _mm256_slli_epi32(y2, 16);
  temp1 = _mm256_and_si256(y3, mask_lo);
  temp2 = _mm256_srli_epi32(y3, 16);
  temp3 = _mm256_and_si256(y2, mask_hi);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y2, xQ_avx);
  temp1 = _mm256_sub_epi16(temp0, y3);
  y2 = _mm256_add_epi16(y2, y3);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 16));
  MReduce(y3, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_epi32(y2, 16);
  temp1 = _mm256_and_si256(y3, mask_hi);
  temp2 = _mm256_slli_epi32(y3, 16);
  temp3 = _mm256_and_si256(y2, mask_lo);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);

  y4 = _mm256_loadu_si256((__m256i *)(p + 64));
  y5 = _mm256_loadu_si256((__m256i *)(p + 80));
  temp0 = _mm256_slli_epi32(y4, 16);
  temp1 = _mm256_and_si256(y5, mask_lo);
  temp2 = _mm256_srli_epi32(y5, 16);
  temp3 = _mm256_and_si256(y4, mask_hi);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y4, xQ_avx);
  temp1 = _mm256_sub_epi16(temp0, y5);
  y4 = _mm256_add_epi16(y4, y5);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 32));
  MReduce(y5, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_epi32(y4, 16);
  temp1 = _mm256_and_si256(y5, mask_hi);
  temp2 = _mm256_slli_epi32(y5, 16);
  temp3 = _mm256_and_si256(y4, mask_lo);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);

  y7 = _mm256_loadu_si256((__m256i *)(p + 112));
  y6 = _mm256_loadu_si256((__m256i *)(p + 96));
  temp0 = _mm256_slli_epi32(y6, 16);
  temp1 = _mm256_and_si256(y7, mask_lo);
  temp2 = _mm256_srli_epi32(y7, 16);
  temp3 = _mm256_and_si256(y6, mask_hi);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y6, xQ_avx);
  temp1 = _mm256_sub_epi16(temp0, y7);
  y6 = _mm256_add_epi16(y6, y7);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 48));
  MReduce(y7, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_epi32(y6, 16);
  temp1 = _mm256_and_si256(y7, mask_hi);
  temp2 = _mm256_slli_epi32(y7, 16);
  temp3 = _mm256_and_si256(y6, mask_lo);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);

  //

  // Level 1
  mask_hi = _mm256_set_epi32(0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0,
                             0xffffffff, 0);
  mask_lo = _mm256_set_epi32(0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0,
                             0xffffffff);

  temp0 = _mm256_slli_epi64(y0, 32);
  temp1 = _mm256_and_si256(y1, mask_lo);

  temp2 = _mm256_srli_epi64(y1, 32);
  temp3 = _mm256_and_si256(y0, mask_hi);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_add_epi16(y0, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y1);
  y0 = _mm256_add_epi16(y0, y1);

  BReduce(y0, y0, ya);

  // Montgomery Reduce y8 * temp1
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 64));
  MReduce(y1, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_srli_epi64(y0, 32);
  temp1 = _mm256_and_si256(y1, mask_hi);
  temp2 = _mm256_slli_epi64(y1, 32);
  temp3 = _mm256_and_si256(y0, mask_lo);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_slli_epi64(y2, 32);
  temp1 = _mm256_and_si256(y3, mask_lo);
  temp2 = _mm256_srli_epi64(y3, 32);
  temp3 = _mm256_and_si256(y2, mask_hi);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y2, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y3);
  y2 = _mm256_add_epi16(y2, y3);
  BReduce(y2, y2, ya);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 80));
  MReduce(y3, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_epi64(y2, 32);
  temp1 = _mm256_and_si256(y3, mask_hi);
  temp2 = _mm256_slli_epi64(y3, 32);
  temp3 = _mm256_and_si256(y2, mask_lo);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_slli_epi64(y4, 32);
  temp1 = _mm256_and_si256(y5, mask_lo);
  temp2 = _mm256_srli_epi64(y5, 32);
  temp3 = _mm256_and_si256(y4, mask_hi);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y4, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y5);
  y4 = _mm256_add_epi16(y4, y5);
  BReduce(y4, y4, ya);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 96));
  MReduce(y5, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_epi64(y4, 32);
  temp1 = _mm256_and_si256(y5, mask_hi);
  temp2 = _mm256_slli_epi64(y5, 32);
  temp3 = _mm256_and_si256(y4, mask_lo);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_slli_epi64(y6, 32);
  temp1 = _mm256_and_si256(y7, mask_lo);
  temp2 = _mm256_srli_epi64(y7, 32);
  temp3 = _mm256_and_si256(y6, mask_hi);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y6, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y7);
  y6 = _mm256_add_epi16(y6, y7);
  BReduce(y6, y6, ya);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 112));
  MReduce(y7, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_epi64(y6, 32);
  temp1 = _mm256_and_si256(y7, mask_hi);
  temp2 = _mm256_slli_epi64(y7, 32);
  temp3 = _mm256_and_si256(y6, mask_lo);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);

  // Level 2

  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 128));
  mask_hi = _mm256_set_epi32(0xffffffff, 0xffffffff, 0, 0, 0xffffffff,
                             0xffffffff, 0, 0);
  mask_lo = _mm256_set_epi32(0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff,
                             0xffffffff);

  temp0 = _mm256_slli_si256(y0, 8);
  temp1 = _mm256_and_si256(y1, mask_lo);
  temp2 = _mm256_srli_si256(y1, 8);
  temp3 = _mm256_and_si256(y0, mask_hi);
  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_add_epi16(y0, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y1);
  y0 = _mm256_add_epi16(y0, y1);

  MReduce(y1, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_srli_si256(y0, 8);
  temp1 = _mm256_and_si256(y1, mask_hi);
  temp2 = _mm256_slli_si256(y1, 8);
  temp3 = _mm256_and_si256(y0, mask_lo);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_slli_si256(y2, 8);
  temp1 = _mm256_and_si256(y3, mask_lo);
  temp2 = _mm256_srli_si256(y3, 8);
  temp3 = _mm256_and_si256(y2, mask_hi);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y2, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y3);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 144));
  y2 = _mm256_add_epi16(y2, y3);
  MReduce(y3, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_si256(y2, 8);
  temp1 = _mm256_and_si256(y3, mask_hi);
  temp2 = _mm256_slli_si256(y3, 8);
  temp3 = _mm256_and_si256(y2, mask_lo);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_slli_si256(y4, 8);
  temp1 = _mm256_and_si256(y5, mask_lo);
  temp2 = _mm256_srli_si256(y5, 8);
  temp3 = _mm256_and_si256(y4, mask_hi);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y4, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y5);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 160));
  y4 = _mm256_add_epi16(y4, y5);
  MReduce(y5, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_si256(y4, 8);
  temp1 = _mm256_and_si256(y5, mask_hi);
  temp2 = _mm256_slli_si256(y5, 8);
  temp3 = _mm256_and_si256(y4, mask_lo);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);

  temp0 = _mm256_slli_si256(y6, 8);
  temp1 = _mm256_and_si256(y7, mask_lo);
  temp2 = _mm256_srli_si256(y7, 8);
  temp3 = _mm256_and_si256(y6, mask_hi);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);
  temp0 = _mm256_add_epi16(y6, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y7);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 176));
  y6 = _mm256_add_epi16(y6, y7);
  MReduce(y7, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_srli_si256(y6, 8);
  temp1 = _mm256_and_si256(y7, mask_hi);
  temp2 = _mm256_slli_si256(y7, 8);
  temp3 = _mm256_and_si256(y6, mask_lo);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);

  // Level 3
  //
  temp0 = _mm256_permute2x128_si256(y0, y1, 0x31);
  y0 = _mm256_permute2x128_si256(y0, y1, 0x20);
  y1 = temp0;
  temp0 = _mm256_add_epi16(y0, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y1);
  y0 = _mm256_add_epi16(y0, y1);
  BReduce(y0, y0, ya);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 192));
  MReduce(y1, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_permute2x128_si256(y0, y1, 0x31);
  y0 = _mm256_permute2x128_si256(y0, y1, 0x20);
  y1 = temp0;

  temp0 = _mm256_permute2x128_si256(y2, y3, 0x31);
  y2 = _mm256_permute2x128_si256(y2, y3, 0x20);
  y3 = temp0;
  temp0 = _mm256_add_epi16(y2, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y3);
  y2 = _mm256_add_epi16(y2, y3);
  BReduce(y2, y2, ya);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 208));
  MReduce(y3, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_permute2x128_si256(y2, y3, 0x31);
  y2 = _mm256_permute2x128_si256(y2, y3, 0x20);
  y3 = temp0;

  //
  //
  //

  temp0 = _mm256_permute2x128_si256(y4, y5, 0x31);
  y4 = _mm256_permute2x128_si256(y4, y5, 0x20);
  y5 = temp0;
  temp0 = _mm256_add_epi16(y4, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y5);
  y4 = _mm256_add_epi16(y4, y5);
  BReduce(y4, y4, ya);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 224));
  MReduce(y5, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_permute2x128_si256(y4, y5, 0x31);
  y4 = _mm256_permute2x128_si256(y4, y5, 0x20);
  y5 = temp0;

  temp0 = _mm256_permute2x128_si256(y6, y7, 0x31);
  y6 = _mm256_permute2x128_si256(y6, y7, 0x20);
  y7 = temp0;
  temp0 = _mm256_add_epi16(y6, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y7);
  y6 = _mm256_add_epi16(y6, y7);
  BReduce(y6, y6, ya);
  y8 = _mm256_load_si256((__m256i *)(omegas_avx + 240));
  MReduce(y7, temp1, y8, y9, ya, yb, yc, yd, ye, yf);
  temp0 = _mm256_permute2x128_si256(y6, y7, 0x31);
  y6 = _mm256_permute2x128_si256(y6, y7, 0x20);
  y7 = temp0;

  // Level 4
  y8 = _mm256_set1_epi16(2285);
  temp0 = _mm256_add_epi16(y0, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y1);
  y0 = _mm256_add_epi16(y0, y1);
  MReduce(y1, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_set1_epi16(758);
  temp0 = _mm256_add_epi16(y2, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y3);
  y2 = _mm256_add_epi16(y2, y3);

  MReduce(y3, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_set1_epi16(1517);
  temp0 = _mm256_add_epi16(y4, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y5);
  y4 = _mm256_add_epi16(y4, y5);

  MReduce(y5, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_set1_epi16(359);
  temp0 = _mm256_add_epi16(y6, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y7);
  y6 = _mm256_add_epi16(y6, y7);

  MReduce(y7, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  // Level 5
  y8 = _mm256_set1_epi16(2285);
  temp0 = _mm256_add_epi16(y0, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y2);
  y0 = _mm256_add_epi16(y0, y2);
  BReduce(y0, y0, ya);

  MReduce(y2, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_add_epi16(y1, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y3);
  y1 = _mm256_add_epi16(y1, y3);
  BReduce(y1, y1, ya);
  MReduce(y3, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_set1_epi16(758);
  temp0 = _mm256_add_epi16(y4, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y6);
  y4 = _mm256_add_epi16(y4, y6);
  BReduce(y4, y4, ya);
  MReduce(y6, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_add_epi16(y5, Q4_avx);
  temp1 = _mm256_sub_epi16(temp0, y7);
  y5 = _mm256_add_epi16(y5, y7);
  BReduce(y5, y5, ya);
  MReduce(y7, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  // Level 6
  y8 = _mm256_set1_epi16(2285);
  temp0 = _mm256_add_epi16(y0, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y4);
  y0 = _mm256_add_epi16(y0, y4);
  MReduce(y4, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_add_epi16(y1, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y5);
  y1 = _mm256_add_epi16(y1, y5);
  MReduce(y5, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_add_epi16(y2, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y6);
  y2 = _mm256_add_epi16(y2, y6);
  MReduce(y6, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  temp0 = _mm256_add_epi16(y3, Q2_avx);
  temp1 = _mm256_sub_epi16(temp0, y7);
  y3 = _mm256_add_epi16(y3, y7);
  MReduce(y7, temp1, y8, y9, ya, yb, yc, yd, ye, yf);

  // Montgomery Reduce all
  y8 = _mm256_load_si256((__m256i *)zetas_inv);
  MReduce(y0, y0, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_load_si256((__m256i *)(zetas_inv + 16));
  MReduce(y1, y1, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_load_si256((__m256i *)(zetas_inv + 32));
  MReduce(y2, y2, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_load_si256((__m256i *)(zetas_inv + 48));
  MReduce(y3, y3, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_load_si256((__m256i *)(zetas_inv + 64));
  MReduce(y4, y4, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_load_si256((__m256i *)(zetas_inv + 80));
  MReduce(y5, y5, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_load_si256((__m256i *)(zetas_inv + 96));
  MReduce(y6, y6, y8, y9, ya, yb, yc, yd, ye, yf);

  y8 = _mm256_load_si256((__m256i *)(zetas_inv + 112));
  MReduce(y7, y7, y8, y9, ya, yb, yc, yd, ye, yf);

  _mm256_storeu_si256((__m256i *)(p + 64), y4);
  _mm256_storeu_si256((__m256i *)(p + 112), y7);
  _mm256_storeu_si256((__m256i *)(p + 96), y6);
  _mm256_storeu_si256((__m256i *)(p + 80), y5);
  _mm256_storeu_si256((__m256i *)(p + 32), y2);
  _mm256_storeu_si256((__m256i *)(p + 48), y3);

  _mm256_storeu_si256((__m256i *)(p), y0);
  _mm256_storeu_si256((__m256i *)(p + 16), y1);
}

void poly_invntt(uint16_t *a) {
  int start, j, jTwiddle, level;
  uint16_t temp, W;
  uint32_t t;

  for (start = 0; start < 1; start++) {
    jTwiddle = 0;
    for (j = start; j < 127; j += 2) {
      W = omegas_inv_bitrev[jTwiddle++];
      temp = a[j];
      a[j] = temp + a[j + 1];
      t = W * ((uint32_t)temp + Q - a[j + 1]);
      a[j + 1] = montgomery_reduce(t);
    }
  }

  for (start = 0; start < 2; start++) {
    jTwiddle = 0;
    for (j = start; j < 127; j += 4) {
      W = omegas_inv_bitrev[jTwiddle++];
      temp = a[j];
      a[j] = barrett_reduce(temp + a[j + 2]);
      t = W * ((uint32_t)temp + (Q << 1) - a[j + 2]);
      a[j + 2] = montgomery_reduce(t);
    }
  }

  for (start = 0; start < 4; start++) {
    jTwiddle = 0;
    for (j = start; j < 127; j += 8) {
      W = omegas_inv_bitrev[jTwiddle++];
      temp = a[j];

      a[j] = temp + a[j + 4];

      t = W * ((uint32_t)temp + (Q << 1) - a[j + 4]);
      a[j + 4] = montgomery_reduce(t);
    }
  }

  for (start = 0; start < 8; start++) {
    jTwiddle = 0;
    for (j = start; j < 127; j += 16) {
      W = omegas_inv_bitrev[jTwiddle++];
      temp = a[j];
      a[j] = barrett_reduce(temp + a[j + 8]);
      t = W * ((uint32_t)temp + (Q << 2) - a[j + 8]);
      a[j + 8] = montgomery_reduce(t);
    }
  }

  for (start = 0; start < 16; start++) {
    jTwiddle = 0;
    for (j = start; j < 127; j += 32) {
      W = omegas_inv_bitrev[jTwiddle++];
      temp = a[j];
      a[j] = temp + a[j + 16];
      t = W * ((uint32_t)temp + (Q << 1) - a[j + 16]);
      a[j + 16] = montgomery_reduce(t);
    }
  }

  for (start = 0; start < 32; start++) {
    jTwiddle = 0;
    for (j = start; j < 127; j += 64) {
      W = omegas_inv_bitrev[jTwiddle++];
      temp = a[j];
      a[j] = barrett_reduce(temp + a[j + 32]);
      t = W * ((uint32_t)temp + (Q << 2) - a[j + 32]);
      a[j + 32] = montgomery_reduce(t);
    }
  }

  for (start = 0; start < 64; start++) {
    jTwiddle = 0;
    for (j = start; j < 127; j += 128) {
      W = omegas_inv_bitrev[jTwiddle++];
      temp = a[j];
      a[j] = temp + a[j + 64];
      t = W * ((uint32_t)temp + (Q << 1) - a[j + 64]);
      a[j + 64] = montgomery_reduce(t);
    }
  }

  for (j = 0; j < 128; j++)
    a[j] = montgomery_reduce(a[j] * zetas_inv[j]);
}

void poly_ntt(uint16_t *p) {
  int level, start, j, kk;
  uint16_t zeta, t;
  kk = 1;
  uint16_t ttt[128];

  for (start = 0; start < 128; start = j + 64) {
    zeta = zetas[kk++];
    for (j = start; j < start + 64; ++j) {

      t = montgomery_reduce((uint32_t)zeta * p[j + 64]);
      p[j + 64] = p[j] + (Q << 1) - t;
      p[j] = p[j] + t;
    }
  }

  for (start = 0; start < 128; start = j + 32) {
    zeta = zetas[kk++];
    for (j = start; j < start + 32; ++j) {
      t = montgomery_reduce((uint32_t)zeta * p[j + 32]);
      p[j + 32] = p[j] + (Q << 1) - t;
      p[j] = (p[j] + t);
    }
  }

  for (start = 0; start < 128; start = j + 16) {
    zeta = zetas[kk++];
    for (j = start; j < start + 16; ++j) {
      t = montgomery_reduce((uint32_t)zeta * p[j + 16]);
      p[j + 16] = p[j] + (Q << 1) - t;
      p[j] = p[j] + t;
    }
  }

  for (start = 0; start < 128; start = j + 8) {
    zeta = zetas[kk++];
    for (j = start; j < start + 8; ++j) {
      t = montgomery_reduce((uint32_t)zeta * p[j + 8]);
      p[j + 8] = p[j] + (Q << 1) - t;
      p[j] = (p[j] + t);
    }
  }

  for (start = 0; start < 128; start = j + 4) {
    zeta = zetas[kk++];
    for (j = start; j < start + 4; ++j) {
      t = montgomery_reduce((uint32_t)zeta * p[j + 4]);
      p[j + 4] = p[j] + (Q << 1) - t;

      p[j] = p[j] + t;
    }
  }

  for (start = 0; start < 128; start = j + 2) {
    zeta = zetas[kk++];
    for (j = start; j < start + 2; ++j) {

      t = montgomery_reduce((uint32_t)zeta * p[j + 2]);
      p[j + 2] = p[j] + (Q << 1) - t;
      p[j] = (p[j] + t);
    }
  }

  for (start = 0; start < 128; start = j + 1) {
    zeta = zetas[kk++];
    for (j = start; j < start + 1; ++j) {

      t = montgomery_reduce((uint32_t)zeta * p[j + 1]);
      p[j + 1] = barrett_reduce(p[j] + (Q << 1) - t);
      p[j] = barrett_reduce(p[j] + t);
    }
  }
}

void poly_ntt_avx(uint16_t *p) {
  int level, start, j, kk;
  uint16_t zeta, t;
  kk = 1;

  __m256i xQ_avx = _mm256_set1_epi16(Q);
  __m256i mask_all = _mm256_set1_epi16(0xffff);
  __m256i qinv_avx = _mm256_set1_epi16(qinv);
  __m256i Q2_avx = _mm256_set1_epi16(Q * 2);
  __m256i BC_avx = _mm256_set1_epi16(BCON);

  __m256i y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf;

  __m256i temp0, temp1, temp2, temp3;

  // Level 0
  y0 = _mm256_loadu_si256((__m256i *)p);
  y8 = _mm256_set1_epi16(2571);
  y4 = _mm256_loadu_si256((__m256i *)(p + 64));
  MReduce(yd, y4, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y0);
  y0 = _mm256_add_epi16(y0, yd);
  y4 = _mm256_sub_epi16(ye, yd);

  y1 = _mm256_loadu_si256((__m256i *)(p + 16));
  y5 = _mm256_loadu_si256((__m256i *)(p + 80));
  MReduce(yd, y5, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y1);
  y1 = _mm256_add_epi16(y1, yd);
  y5 = _mm256_sub_epi16(ye, yd);

  y2 = _mm256_loadu_si256((__m256i *)(p + 32));
  y6 = _mm256_loadu_si256((__m256i *)(p + 96));
  MReduce(yd, y6, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y2);
  y2 = _mm256_add_epi16(y2, yd);
  y6 = _mm256_sub_epi16(ye, yd);

  y3 = _mm256_loadu_si256((__m256i *)(p + 48));
  y7 = _mm256_loadu_si256((__m256i *)(p + 112));
  MReduce(yd, y7, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y3);
  y3 = _mm256_add_epi16(y3, yd);
  y7 = _mm256_sub_epi16(ye, yd);

  // Level 1
  y8 = _mm256_set1_epi16(2970);

  //(y0,y2)(y1,y3)
  MReduce(yd, y2, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y0);
  y0 = _mm256_add_epi16(y0, yd);
  y2 = _mm256_sub_epi16(ye, yd);

  MReduce(yd, y3, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y1);
  y1 = _mm256_add_epi16(y1, yd);
  y3 = _mm256_sub_epi16(ye, yd);

  y8 = _mm256_set1_epi16(1812);
  //(y4,y6)(y5,y7)
  MReduce(yd, y6, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y4);
  y4 = _mm256_add_epi16(y4, yd);
  y6 = _mm256_sub_epi16(ye, yd);

  MReduce(yd, y7, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y5);
  y5 = _mm256_add_epi16(y5, yd);
  y7 = _mm256_sub_epi16(ye, yd);

  // Level 2
  //(y0,y1)
  y8 = _mm256_set1_epi16(1493);
  MReduce(yd, y1, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y0);
  y0 = _mm256_add_epi16(y0, yd);
  y1 = _mm256_sub_epi16(ye, yd);

  //(y2,y3)
  y8 = _mm256_set1_epi16(1422);
  MReduce(yd, y3, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y2);
  y2 = _mm256_add_epi16(y2, yd);
  y3 = _mm256_sub_epi16(ye, yd);

  //(y4,y5)
  y8 = _mm256_set1_epi16(287);
  MReduce(yd, y5, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y4);
  y4 = _mm256_add_epi16(y4, yd);
  y5 = _mm256_sub_epi16(ye, yd);

  //(y6,y7)
  y8 = _mm256_set1_epi16(202);
  MReduce(yd, y7, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y6);
  y6 = _mm256_add_epi16(y6, yd);
  y7 = _mm256_sub_epi16(ye, yd);

  // Level 3
  y8 = _mm256_load_si256((__m256i *)zetas_avx);
  // compute y0, y1 together
  temp0 = _mm256_permute2x128_si256(y0, y1, 0x31);
  y0 = _mm256_permute2x128_si256(y0, y1, 0x20);
  y1 = temp0;

  MReduce(yd, y1, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y0);
  y0 = _mm256_add_epi16(y0, yd);
  y1 = _mm256_sub_epi16(ye, yd);

  temp0 = _mm256_permute2x128_si256(y0, y1, 0x31);
  y0 = _mm256_permute2x128_si256(y0, y1, 0x20);
  y1 = temp0;

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 16));
  temp0 = _mm256_permute2x128_si256(y2, y3, 0x31);
  y2 = _mm256_permute2x128_si256(y2, y3, 0x20);
  y3 = temp0;

  MReduce(yd, y3, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y2);
  y2 = _mm256_add_epi16(y2, yd);
  y3 = _mm256_sub_epi16(ye, yd);

  temp0 = _mm256_permute2x128_si256(y2, y3, 0x31);
  y2 = _mm256_permute2x128_si256(y2, y3, 0x20);
  y3 = temp0;

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 32));
  temp0 = _mm256_permute2x128_si256(y4, y5, 0x31);
  y4 = _mm256_permute2x128_si256(y4, y5, 0x20);
  y5 = temp0;

  MReduce(yd, y5, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y4);
  y4 = _mm256_add_epi16(y4, yd);
  y5 = _mm256_sub_epi16(ye, yd);

  temp0 = _mm256_permute2x128_si256(y4, y5, 0x31);
  y4 = _mm256_permute2x128_si256(y4, y5, 0x20);
  y5 = temp0;

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 48));
  temp0 = _mm256_permute2x128_si256(y6, y7, 0x31);
  y6 = _mm256_permute2x128_si256(y6, y7, 0x20);
  y7 = temp0;

  MReduce(yd, y7, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y6);
  y6 = _mm256_add_epi16(y6, yd);
  y7 = _mm256_sub_epi16(ye, yd);

  temp0 = _mm256_permute2x128_si256(y6, y7, 0x31);
  y6 = _mm256_permute2x128_si256(y6, y7, 0x20);
  y7 = temp0;

  // Level 4
  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 64));
  __m256i mask_hi = _mm256_set_epi32(0xffffffff, 0xffffffff, 0, 0, 0xffffffff,
                                     0xffffffff, 0, 0);
  __m256i mask_lo = _mm256_set_epi32(0, 0, 0xffffffff, 0xffffffff, 0, 0,
                                     0xffffffff, 0xffffffff);

  temp0 = _mm256_slli_si256(y0, 8);
  temp1 = _mm256_and_si256(y1, mask_lo);
  temp2 = _mm256_srli_si256(y1, 8);
  temp3 = _mm256_and_si256(y0, mask_hi);
  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  MReduce(yd, y1, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y0);
  y0 = _mm256_add_epi16(y0, yd);
  y1 = _mm256_sub_epi16(ye, yd);

  temp0 = _mm256_srli_si256(y0, 8);
  temp1 = _mm256_and_si256(y1, mask_hi);
  temp2 = _mm256_slli_si256(y1, 8);
  temp3 = _mm256_and_si256(y0, mask_lo);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 80));
  temp0 = _mm256_slli_si256(y2, 8);
  temp1 = _mm256_and_si256(y3, mask_lo);
  temp2 = _mm256_srli_si256(y3, 8);
  temp3 = _mm256_and_si256(y2, mask_hi);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);
  MReduce(yd, y3, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y2);
  y2 = _mm256_add_epi16(y2, yd);
  y3 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_si256(y2, 8);
  temp1 = _mm256_and_si256(y3, mask_hi);
  temp2 = _mm256_slli_si256(y3, 8);
  temp3 = _mm256_and_si256(y2, mask_lo);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 96));
  temp0 = _mm256_slli_si256(y4, 8);
  temp1 = _mm256_and_si256(y5, mask_lo);
  temp2 = _mm256_srli_si256(y5, 8);
  temp3 = _mm256_and_si256(y4, mask_hi);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);
  MReduce(yd, y5, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y4);
  y4 = _mm256_add_epi16(y4, yd);
  y5 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_si256(y4, 8);
  temp1 = _mm256_and_si256(y5, mask_hi);
  temp2 = _mm256_slli_si256(y5, 8);
  temp3 = _mm256_and_si256(y4, mask_lo);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 112));
  temp0 = _mm256_slli_si256(y6, 8);
  temp1 = _mm256_and_si256(y7, mask_lo);
  temp2 = _mm256_srli_si256(y7, 8);
  temp3 = _mm256_and_si256(y6, mask_hi);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);
  MReduce(yd, y7, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y6);
  y6 = _mm256_add_epi16(y6, yd);
  y7 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_si256(y6, 8);
  temp1 = _mm256_and_si256(y7, mask_hi);
  temp2 = _mm256_slli_si256(y7, 8);
  temp3 = _mm256_and_si256(y6, mask_lo);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);

  // Level 5
  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 128));
  mask_hi = _mm256_set_epi32(0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0,
                             0xffffffff, 0);
  mask_lo = _mm256_set_epi32(0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0,
                             0xffffffff);
  temp0 = _mm256_slli_epi64(y0, 32);
  temp1 = _mm256_and_si256(y1, mask_lo);
  temp2 = _mm256_srli_epi64(y1, 32);
  temp3 = _mm256_and_si256(y0, mask_hi);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  MReduce(yd, y1, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y0);
  y0 = _mm256_add_epi16(y0, yd);
  y1 = _mm256_sub_epi16(ye, yd);

  temp0 = _mm256_srli_epi64(y0, 32);
  temp1 = _mm256_and_si256(y1, mask_hi);
  temp2 = _mm256_slli_epi64(y1, 32);
  temp3 = _mm256_and_si256(y0, mask_lo);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 144));
  temp0 = _mm256_slli_epi64(y2, 32);
  temp1 = _mm256_and_si256(y3, mask_lo);
  temp2 = _mm256_srli_epi64(y3, 32);
  temp3 = _mm256_and_si256(y2, mask_hi);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);
  MReduce(yd, y3, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y2);
  y2 = _mm256_add_epi16(y2, yd);
  y3 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_epi64(y2, 32);
  temp1 = _mm256_and_si256(y3, mask_hi);
  temp2 = _mm256_slli_epi64(y3, 32);
  temp3 = _mm256_and_si256(y2, mask_lo);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 160));
  temp0 = _mm256_slli_epi64(y4, 32);
  temp1 = _mm256_and_si256(y5, mask_lo);
  temp2 = _mm256_srli_epi64(y5, 32);
  temp3 = _mm256_and_si256(y4, mask_hi);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);
  MReduce(yd, y5, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y4);
  y4 = _mm256_add_epi16(y4, yd);
  y5 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_epi64(y4, 32);
  temp1 = _mm256_and_si256(y5, mask_hi);
  temp2 = _mm256_slli_epi64(y5, 32);
  temp3 = _mm256_and_si256(y4, mask_lo);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 176));
  temp0 = _mm256_slli_epi64(y6, 32);
  temp1 = _mm256_and_si256(y7, mask_lo);
  temp2 = _mm256_srli_epi64(y7, 32);
  temp3 = _mm256_and_si256(y6, mask_hi);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);
  MReduce(yd, y7, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y6);
  y6 = _mm256_add_epi16(y6, yd);
  y7 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_epi64(y6, 32);
  temp1 = _mm256_and_si256(y7, mask_hi);
  temp2 = _mm256_slli_epi64(y7, 32);
  temp3 = _mm256_and_si256(y6, mask_lo);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);

  // Level 6
  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 192));
  mask_hi = _mm256_set1_epi32(0xffff0000);
  mask_lo = _mm256_set1_epi32(0xffff);
  temp0 = _mm256_slli_epi32(y0, 16);
  temp1 = _mm256_and_si256(y1, mask_lo);
  temp2 = _mm256_srli_epi32(y1, 16);
  temp3 = _mm256_and_si256(y0, mask_hi);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  MReduce(yd, y1, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y0);
  y0 = _mm256_add_epi16(y0, yd);
  y1 = _mm256_sub_epi16(ye, yd);

  temp0 = _mm256_srli_epi32(y0, 16);
  temp1 = _mm256_and_si256(y1, mask_hi);
  temp2 = _mm256_slli_epi32(y1, 16);
  temp3 = _mm256_and_si256(y0, mask_lo);

  y0 = _mm256_xor_si256(temp0, temp1);
  y1 = _mm256_xor_si256(temp2, temp3);

  BReduce(y0, y0, ya);
  BReduce(y1, y1, yb);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 208));
  temp0 = _mm256_slli_epi32(y2, 16);
  temp1 = _mm256_and_si256(y3, mask_lo);
  temp2 = _mm256_srli_epi32(y3, 16);
  temp3 = _mm256_and_si256(y2, mask_hi);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);
  MReduce(yd, y3, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y2);
  y2 = _mm256_add_epi16(y2, yd);
  y3 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_epi32(y2, 16);
  temp1 = _mm256_and_si256(y3, mask_hi);
  temp2 = _mm256_slli_epi32(y3, 16);
  temp3 = _mm256_and_si256(y2, mask_lo);
  y2 = _mm256_xor_si256(temp0, temp1);
  y3 = _mm256_xor_si256(temp2, temp3);
  BReduce(y2, y2, ya);
  BReduce(y3, y3, yb);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 224));
  temp0 = _mm256_slli_epi32(y4, 16);
  temp1 = _mm256_and_si256(y5, mask_lo);
  temp2 = _mm256_srli_epi32(y5, 16);
  temp3 = _mm256_and_si256(y4, mask_hi);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);

  MReduce(yd, y5, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y4);
  y4 = _mm256_add_epi16(y4, yd);
  y5 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_epi32(y4, 16);
  temp1 = _mm256_and_si256(y5, mask_hi);
  temp2 = _mm256_slli_epi32(y5, 16);
  temp3 = _mm256_and_si256(y4, mask_lo);
  y4 = _mm256_xor_si256(temp0, temp1);
  y5 = _mm256_xor_si256(temp2, temp3);
  BReduce(y4, y4, ya);
  BReduce(y5, y5, yb);

  y8 = _mm256_load_si256((__m256i *)(zetas_avx + 240));
  temp0 = _mm256_slli_epi32(y6, 16);
  temp1 = _mm256_and_si256(y7, mask_lo);
  temp2 = _mm256_srli_epi32(y7, 16);
  temp3 = _mm256_and_si256(y6, mask_hi);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);

  MReduce(yd, y7, y8, y9, ya, yb, yc, yd, ye, yf);
  ye = _mm256_add_epi16(Q2_avx, y6);
  y6 = _mm256_add_epi16(y6, yd);
  y7 = _mm256_sub_epi16(ye, yd);
  temp0 = _mm256_srli_epi32(y6, 16);
  temp1 = _mm256_and_si256(y7, mask_hi);
  temp2 = _mm256_slli_epi32(y7, 16);
  temp3 = _mm256_and_si256(y6, mask_lo);
  y6 = _mm256_xor_si256(temp0, temp1);
  y7 = _mm256_xor_si256(temp2, temp3);
  BReduce(y6, y6, ya);
  BReduce(y7, y7, yb);

  _mm256_storeu_si256((__m256i *)(p + 64), y4);
  _mm256_storeu_si256((__m256i *)(p + 112), y7);
  _mm256_storeu_si256((__m256i *)(p + 96), y6);
  _mm256_storeu_si256((__m256i *)(p + 80), y5);
  _mm256_storeu_si256((__m256i *)(p + 32), y2);
  _mm256_storeu_si256((__m256i *)(p + 48), y3);
  _mm256_storeu_si256((__m256i *)(p), y0);
  _mm256_storeu_si256((__m256i *)(p + 16), y1);

  return;
}

void kntt_avx(uint16_t *r) {
  poly_ntt_avx(r);
  poly_ntt_avx(r + 128);
  poly_ntt_avx(r + 256);
  poly_ntt_avx(r + 384);
  poly_ntt_avx(r + 512);
  poly_ntt_avx(r + 640);
  poly_ntt_avx(r + 768);
  poly_ntt_avx(r + 896);
}

void kinv_ntt_avx(uint16_t *r) {
  poly_invntt_avx(r);
  poly_invntt_avx(r + 128);
  poly_invntt_avx(r + 256);
  poly_invntt_avx(r + 384);
  poly_invntt_avx(r + 512);
  poly_invntt_avx(r + 640);
  poly_invntt_avx(r + 768);
  poly_invntt_avx(r + 896);
}

void kntt(uint16_t *r) {
  poly_ntt(r);
  poly_ntt(r + 128);
  poly_ntt(r + 256);
  poly_ntt(r + 384);
  poly_ntt(r + 512);
  poly_ntt(r + 640);
  poly_ntt(r + 768);
  poly_ntt(r + 896);
}

void kinv_ntt(uint16_t *r) {
  poly_invntt(r);
  poly_invntt(r + 128);
  poly_invntt(r + 256);
  poly_invntt(r + 384);
  poly_invntt(r + 512);
  poly_invntt(r + 640);
  poly_invntt(r + 768);
  poly_invntt(r + 896);
}

int test_ntt() {
  LARGE_INTEGER start, end;
  LARGE_INTEGER frequency;

  uint16_t a[1024], b[1024], c[1024];
  uint16_t ta[1024], tb[1024], tc[1024];

  int i, j;
  double elapsed;

  QueryPerformanceFrequency(&frequency);

  double quadpart = (double)frequency.QuadPart;

  srand(time(NULL));
  for (i = 0; i < 1024; i++) {
    a[i] = b[i] = 0;
  }
  a[0] = 1;
  a[1023] = 1;
  b[1] = 1;

  kntt(a);

  kntt(b);

  poly_pointwise(c, b, a);

  /*for (i = 0; i < 1024; i++)
  {
          if (c[i] % 3329 != 0)
          {
                  printf("deg = %d, coeff = %d\n", i, c[i] % 3329);
          }
  }
  printf("\n\n\n\n");*/

  kinv_ntt(c);

  for (i = 0; i < 1024; i++) {
    if (c[i] % 3329 != 0) {
      printf("deg = %d, coeff = %d\n", i, c[i] % 3329);
    }
  }

  for (i = 0; i < 1000; i++) {
    for (j = 0; j < 1024; j++) {
      a[j] = rand() % Q;
      ta[j] = a[j];
      b[j] = rand() % Q;
      tb[j] = b[j];
    }
    poly_pointwise(c, b, a);

    poly_pointwise_avx(tc, tb, ta);

    /*for (j = 0; j < 128; j++)
    {
            printf("%d, ", tc[j]);
    }
    printf("\n");

    for (j = 0; j < 128; j++)
    {
            printf("%d, ", c[j]);
    }
    printf("\n");
    printf("\n\n\n");*/

    for (j = 0; j < 1024; j++) {
      if (c[j] != tc[j]) {
        /*printf("%d\n", j);*/
        return -3;
      }
    }
  }

  return -100;

  for (j = 0; j < 1024; j++) {
    a[j] = (Q + ((rand()) % 3) * 2 - 2) % Q;
    b[j] = a[j];
  }

  QueryPerformanceCounter(&start);
#if test_time
  for (j = 0; j < 1000000; j++)
#else
  for (j = 0; j < 1; j++)
#endif
  {
    poly_ntt(a);
    poly_ntt(a + 128);
    poly_ntt(a + 256);
    poly_ntt(a + 384);
    poly_ntt(a + 512);
    poly_ntt(a + 640);
    poly_ntt(a + 768);
    poly_ntt(a + 896);
  }
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;

#if test_time
  printf("Original Forward NTT RUNS in Time : %0.6f usecs\n ", elapsed);
#endif

  QueryPerformanceCounter(&start);
#if test_time
  for (j = 0; j < 1000000; j++)
#else
  for (j = 0; j < 1; j++)
#endif
  {
    poly_ntt_avx(b);
    poly_ntt_avx(b + 128);
    poly_ntt_avx(b + 256);
    poly_ntt_avx(b + 384);
    poly_ntt_avx(b + 512);
    poly_ntt_avx(b + 640);
    poly_ntt_avx(b + 768);
    poly_ntt_avx(b + 896);
  }

  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;

#if test_time
  printf("AVX Forward NTT in Time : %0.6f usecs\n\n", elapsed);
#endif

  for (j = 0; j < 1024; j++) {
    a[j] = rand() % (Q);
    b[j] = rand() % (Q);
    c[j] = 0;

    ta[j] = a[j];
    tb[j] = b[j];
    tc[j] = c[j];
  }

  QueryPerformanceCounter(&start);
#if test_time
  for (j = 0; j < 1000000; j++)
#else
  for (j = 0; j < 1; j++)
#endif
  {
    poly_pointwise(c, b, a);
  }
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;
  // printf("%d\n", c[0]);
#if test_time
  printf("Original PonitMult  RUNS in Time : %0.6f usecs\n ", elapsed);
#endif

  QueryPerformanceCounter(&start);
#if test_time
  for (j = 0; j < 1000000; j++)
#else
  for (j = 0; j < 1; j++)
#endif
  {
    poly_pointwise_avx(tc, tb, ta);
  }

  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;

#if test_time
  printf("AVX PonitMult RUNS in Time : %0.6f usecs\n\n", elapsed);
#endif

  for (j = 0; j < 1024; j++) {
    a[j] = rand() % Q;
    b[j] = a[j];
  }

  QueryPerformanceCounter(&start);

#if test_time
  for (j = 0; j < 1000000; j++)
#else
  for (j = 0; j < 1; j++)
#endif
  {
    poly_invntt(a);
    poly_invntt(a + 128);
    poly_invntt(a + 256);
    poly_invntt(a + 384);
    poly_invntt(a + 512);
    poly_invntt(a + 640);
    poly_invntt(a + 768);
    poly_invntt(a + 896);
  }
  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;

#if test_time
  printf("Original Backward NTT RUNS in Time : %0.6f usecs\n ", elapsed);
#endif

  QueryPerformanceCounter(&start);
#if test_time
  for (j = 0; j < 1000000; j++)
#else
  for (j = 0; j < 1; j++)
#endif
  {
    poly_invntt_avx(b);
    poly_invntt_avx(b + 128);
    poly_invntt_avx(b + 256);
    poly_invntt_avx(b + 384);
    poly_invntt_avx(b + 512);
    poly_invntt_avx(b + 640);
    poly_invntt_avx(b + 768);
    poly_invntt_avx(b + 896);
  }

  QueryPerformanceCounter(&end);
  elapsed = (end.QuadPart - start.QuadPart) / quadpart;

#if test_time
  printf("AVX Backward NTT in Time : %0.6f usecs\n\n", elapsed);
#endif

  for (i = 0; i < 100000; i++) {
    for (j = 0; j < 1024; j++) {
      a[j] = (Q + ((rand()) % 3) * 2 - 2) % Q;
      b[j] = a[j];
    }
    poly_ntt_avx(b);
    poly_ntt_avx(b + 128);
    poly_ntt_avx(b + 256);
    poly_ntt_avx(b + 384);
    poly_ntt_avx(b + 512);
    poly_ntt_avx(b + 640);
    poly_ntt_avx(b + 768);
    poly_ntt_avx(b + 896);

    poly_ntt(a);
    poly_ntt(a + 128);
    poly_ntt(a + 256);
    poly_ntt(a + 384);
    poly_ntt(a + 512);
    poly_ntt(a + 640);
    poly_ntt(a + 768);
    poly_ntt(a + 896);

    for (j = 0; j < 1024; j++) {
      if (a[j] != b[j]) {
        return -1;
      }
    }
  }

  for (i = 0; i < 100000; i++) {
    for (j = 0; j < 1024; j++) {
      a[j] = rand() % Q;
      b[j] = a[j];
    }
    poly_invntt(a);
    poly_invntt(a + 128);
    poly_invntt(a + 256);
    poly_invntt(a + 384);
    poly_invntt(a + 512);
    poly_invntt(a + 640);
    poly_invntt(a + 768);
    poly_invntt(a + 896);

    poly_invntt_avx(b);
    poly_invntt_avx(b + 128);
    poly_invntt_avx(b + 256);
    poly_invntt_avx(b + 384);
    poly_invntt_avx(b + 512);
    poly_invntt_avx(b + 640);
    poly_invntt_avx(b + 768);
    poly_invntt_avx(b + 896);

    for (j = 0; j < 1024; j++) {
      if (a[j] != b[j]) {
        return -2;
      }
    }
  }

  return 0;
}
