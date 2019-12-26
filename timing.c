#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>

#include "ntt.h"
#include "poly.h"

#include "parameter.h"

int main() {
  int ret_val;

  // ret_val = test_poly();
  // if (ret_val == -1)
  //{
  //	printf("poly2byte wrong !!!\n\n\n");
  //}
  // else if (ret_val == -2)
  //{
  //	printf("Byte2Poly AVX  WRONG!!!\n\n\n");
  //}
  // else if (ret_val == -3)
  //{
  //	printf("Cannot be inverted!!!\n\n\n");
  //}
  // else if (ret_val == 0)
  //{
  //	printf("POLY AVX  Right!!!\n\n\n");
  //}
  // else
  //{
  //	printf("something in POLY is Wrong!!!\n\n\n");
  //}

  ret_val = test_ntt();
  if (ret_val == -1) {
    printf("Forward NTT AVX  WRONG!!!\n\n\n");
  } else if (ret_val == -2) {
    printf("Backward NTT AVX  WRONG!!!\n\n\n");
  } else if (ret_val == -3) {
    printf("PointMult AVX  WRONG!!!\n\n\n");
  } else if (ret_val == 0) {
    printf("NTT AVX  Right!!!\n\n\n");
  } else {
    printf("Somewhere is not  Right!!!\n\n\n");
  }

  //	test_CBD();

  system("pause");
  return 0;
}