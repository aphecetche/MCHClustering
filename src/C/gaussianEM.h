# ifndef  _GAUSSIANEM_H
# define  _GAUSSIANEM_H

extern "C" {
  void computeDiscretizedGaussian2D( const double *xyInfSup, const double *theta, 
                                   int K, int N, int k, double *z );

  void generateMixedGaussians2D( const double *xyInfSup, const double *theta, 
                               int K, int N, double *z);

  double computeWeightedLogLikelihood( const double *xyInfSup, const double *theta, const double *z, 
                                     int K, int N);

  void weightedEMLoop( const double *xyDxy, const double *z, const double *theta0, int K, int N, 
                    int mode, double LConvergence, int verbose, double *theta);
}

#endif // _GAUSSIANEM_H