#ifndef _MATRIX2D_H_
#define _MATRIX2D_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    size_t m; /* nrows */
    size_t n; /* ncols */
    double *data; /* data[i][j] = data[i*n + j] */
    int refcounts;
}Matrix2D;

Matrix2D MatrixMake(size_t m, size_t n);
Matrix2D MatrixRandom(size_t m, size_t n);
Matrix2D MatrixRead(char *filename, size_t m, size_t n);
Matrix2D MatrixClone(Matrix2D *mat);

void MatrixFree(Matrix2D *mat);
void MatrixShow(Matrix2D *mat);
void MatrixCopy(Matrix2D *mat1, Matrix2D *mat2);
void MatrixSave(Matrix2D *mat, char *filename);


Matrix2D MatrixDot(Matrix2D *mat1, Matrix2D *mat2);
Matrix2D MatrixT(Matrix2D *mat);
Matrix2D MatrixSum(Matrix2D *mat, short axis);
Matrix2D MatrixExpand(Matrix2D *mat, unsigned int copies);
Matrix2D MatrixApply(Matrix2D *mat, double(*pf)(double));
Matrix2D MatrixPow(Matrix2D *mat, int d);
Matrix2D MatrixSoftmax(Matrix2D *mat);

Matrix2D MatrixAddScalar(Matrix2D *mat, double scalar);
Matrix2D MatrixSubScalar(Matrix2D *mat, double scalar);
Matrix2D MatrixMulScalar(Matrix2D *mat, double scalar);
Matrix2D MatrixDivScalar(Matrix2D *mat, double scalar);

Matrix2D MatrixAddMatrix(Matrix2D *mat1, Matrix2D *mat2);
Matrix2D MatrixSubMatrix(Matrix2D *mat1, Matrix2D *mat2);
Matrix2D MatrixMulMatrix(Matrix2D *mat1, Matrix2D *mat2);
Matrix2D MatrixDivMatrix(Matrix2D *mat1, Matrix2D *mat2);

#endif /*_MATRIX2D_H_*/
