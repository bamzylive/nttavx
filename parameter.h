//
//  parameter.h

#ifndef parameter_h
#define parameter_h

#define N 1024
#define Q 3329
//#define k 1  //central binomial

#define N8 128
#define N4 256
#define N2 512

/*
#define PKE_SECRETKEYBYTES 12*N/8
#define PKE_PUBLICKEYBYTES 12*N/8+32
#define PKE_SEED_BYTES 32   // the size of seed
#define PKE_POLY_BYTES  12*N/8 //
#define PKE_MESS_BYTES   N/8
#define PKE_CIP_BYTES   (12*N+2*N)/8
#define PKE_ALGNAME  TALE
*/

#define PKE_SECRETKEYBYTES 1536
#define PKE_PUBLICKEYBYTES 1568
#define PKE_SEED_BYTES 32   // the size of seed
#define PKE_POLY_BYTES 1536 //
#define PKE_MESS_BYTES 128
#define PKE_CIP_BYTES 1792

#define KEM_PUBLICKEYBYTES PKE_PUBLICKEYBYTES
#define KEM_SECRETKEYBYTES                                                     \
  PKE_SECRETKEYBYTES + PKE_PUBLICKEYBYTES + 2 * PKE_SEED_BYTES
#define KEM_CIP_BYTES PKE_CIP_BYTES + PKE_SEED_BYTES
#define KEM_KEYBYTES 32

#define PKE_ALGNAME TALE

#endif /* parameter_h */
