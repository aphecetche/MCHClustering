# ifndef  _MATHUTIL_H
# define  _MATHUTIL_H

# include <math.h>
# include <float.h>
# include <stddef.h>

typedef short Mask_t;


inline static void vectorSetZero( double *u, int N) { 
  for (int i=0; i < N; i++) u[i] = 0; return;}

inline static void vectorSetZeroInt( int *u, int N) { 
  for (int i=0; i < N; i++) u[i] = 0; return;}

inline static void vectorSetZeroShort( short *u, int N) { 
  for (int i=0; i < N; i++) u[i] = 0; return;}

inline static void vectorSet( double *u, double value, int N) { 
  for (int i=0; i < N; i++) u[i] = value; return;}

inline static void vectorSetInt( int *u, int value, int N) { 
  for (int i=0; i < N; i++) u[i] = value; return;}

inline static void vectorSetShort( short *u, short value, int N) { 
  for (int i=0; i < N; i++) u[i] = value; return;}

inline static void vectorCopy( const double *src, int N, double *dest) { 
  for (int i=0; i < N; i++) dest[i] = src[i]; return;}

inline static void vectorCopyShort( const short *src, int N, short *dest) { 
  for (int i=0; i < N; i++) dest[i] = src[i]; return;}

inline static void vectorAddVector( const double *u, double cst, const double *v, int N, double *res) { 
  for (int i=0; i < N; i++) res[i] = u[i] + cst*v[i]; return;}
     
 inline static void vectorAddScalar( const double *u, double cst, int N, double *res) { 
  for (int i=0; i < N; i++) res[i] = u[i] + cst; return;}  
  
inline static void vectorMultVector( const double *u, const double *v, int N, double *res) {
  for (int i=0; i < N; i++) res[i] = u[i] * v[i]; return;}

inline static void vectorMultScalar( const double *u, double cst, int N, double *res) {
  for (int i=0; i < N; i++) res[i] = u[i] * cst; return;}

inline static double vectorSum( const double *u, int N) { 
  double res = 0; for (int i=0; i < N; i++) res += u[i]; return res; }

inline static int vectorSumInt( const int *u, int N) { 
  int res = 0; for (int i=0; i < N; i++) res += u[i]; return res; }

inline static int vectorSumShort( const short *u, int N) { 
  int res = 0; for (int i=0; i < N; i++) res += u[i]; return res; }

inline static int vectorSumRowInt( const int *matrix, int N, int M ) {
  int res = 0; for (int j=0; j < M; j++) res += matrix[j]; return res; }

inline static int vectorSumColumnInt( const int *matrix, int N, int M ) {
  int res = 0; for (int i=0; i < N; i++) res += matrix[i*M]; return res; }

inline static double vectorMin( const double *u, int N) { 
  double res = DBL_MAX; for (int i=0; i < N; i++) res = fmin(res, u[i]); return res; }

inline static double vectorMax( const double *u, int N) { 
  double res = -DBL_MAX; for (int i=0; i < N; i++) res = fmax(res, u[i]); return res; }
//
// Logical operations
//
inline static void vectorNotShort( const short *src, int N, short *dest) { 
  for (int i=0; i < N; i++) dest[i] = ! src[i] ; return;}
//
// Mask operations
//
inline static int vectorBuildMaskEqualShort( short *src, short value, int N, short *mask) {
    int count=0; for(int i=0; i < N; i++) { mask[i] = (src[i] == value); count += ((src[i] == value));} return count;}

inline static int vectorMaskedCopy( const double *v, const Mask_t *mask, int N, double *maskedVector ) { 
  int k=0;for (int i=0; i < N; i++) { if ( mask[i] ) maskedVector[k++] = v[i];} return k;}

inline static double vectorMaskedSum( const double *v, const Mask_t *mask, int N ) { 
  double sum=0; for (int i=0, k=0; i < N; i++) { sum += v[i]*mask[i];} return sum;}

void vectorPrint( const char *str, const double *x, int N);
void vectorPrintInt( const char *str, const int *x, int N);
void vectorPrintShort( const char *str, const short *x, int N);
void vectorPrint2Columns( const char *str, const double *x, const double *y, int N);

#endif // _MATHUTIL_H
