#include "Matrix2D.h"

#include <time.h>


Matrix2D MatrixMake(size_t m, size_t n) {
    Matrix2D mat;
    mat.refcounts = 0;
    mat.m = m; mat.n = n;
    mat.data = (double*) calloc(m*n, sizeof(double));
    return mat;
}

void MatrixFree(Matrix2D *mat) {
    mat->m = 0; mat->n = 0;
    mat->refcounts = 0;
    free(mat->data);
}

Matrix2D MatrixClone(Matrix2D *mat) {
    Matrix2D res = MatrixMake(mat->n, mat->m);
    for(int i=0; i<(int)mat->m; i++) {
        for(int j=0; j<(int)mat->n; j++) {
            res.data[i*res.n + j] = mat->data[i*mat->n + j];}}
    return res;
}

void MatrixShow(Matrix2D *mat) {
    printf("Matrix2D(%lu, %lu)\n", mat->m, mat->n);
    for(int i=0; i<(int)mat->m; i++) {
        for(int j=0; j<(int)mat->n; j++) {
            printf("%12.6lf ", mat->data[i*mat->n + j]);}
        printf("\n");}
    printf("\n");
}

void MatrixSave(Matrix2D *mat, char *filename) {
    FILE *fptr;

    fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Cannot read Matrix file does not exists\n");
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<(int)mat->m; i++){
        for(int j=0; j<(int)mat->n; j++){
            fprintf(fptr, "%lf ", mat->data[i*mat->n + j]);}}
}

void MatrixCopy(Matrix2D *mat1, Matrix2D *mat2) {
    if(mat1->m == mat2->m && mat1->n == mat2->n) {
        for(int i=0; i<(int)mat1->m; i++) {
            for(int j=0; j<(int)mat1->n; j++) {
                mat1->data[i*mat1->n + j] = mat2->data[i*mat2->n + j];;}}}
}

Matrix2D MatrixDot(Matrix2D *mat1, Matrix2D *mat2) {
    Matrix2D res = MatrixMake(mat1->m, mat2->n);
    for(int k=0; k<(int)mat1->n; k++){
        for(int i=0; i<(int)mat1->m; i++){
            for(int j=0; j<(int)mat2->n; j++){
                res.data[i*res.n + j] += mat1->data[i*mat1->n + k] *
                                         mat2->data[k*mat2->n + j];}}}
    return res;
}

Matrix2D MatrixT(Matrix2D *mat) {
    Matrix2D res = MatrixMake(mat->n, mat->m);
    for(int i=0; i<(int)mat->m; i++) {
        for(int j=0; j<(int)mat->n; j++) {
            res.data[j*res.n + i] = mat->data[i*mat->n + j];}}

    return res;
}

Matrix2D MatrixSum(Matrix2D *mat, short axis) {
    Matrix2D res;
    if(axis == 0) {
        res = MatrixMake(1, mat->n);
        for(int i=0; i<(int)mat->m; i++){
            for(int j=0; j<(int)mat->n; j++){
                res.data[j] += mat->data[i*mat->n + j];}}}
    else if (axis == 1) {
        res = MatrixMake(1, mat->m);
        for(int i=0; i<(int)mat->m; i++){
            for(int j=0; j<(int)mat->n; j++){
                res.data[i] += mat->data[i*mat->n + j];}}}
    return res;
}

Matrix2D MatrixPow(Matrix2D *mat, int d) {
    Matrix2D res = MatrixMake(mat->n, mat->m);
    for(int i=0; i<(int)mat->m; i++){
            for(int j=0; j<(int)mat->n; j++){
                res.data[i*res.n + j] = pow(mat->data[i*mat->n + j], d);}}
    return res;
}

Matrix2D MatrixExpand(Matrix2D *mat, unsigned int copies) {
    Matrix2D res;
    if(mat->n == 1) {
        res = MatrixMake(mat->m, copies);
        for(int i=0; i<(int)res.m; i++) {
            for(int j=0; j<(int)res.n; j++) {
                res.data[i*copies + j] = mat->data[i];}}}
    else if(mat->m == 1) {
        res = MatrixMake(copies, mat->n);
        for(int i=0; i<(int)res.m; i++) {
            for(int j=0; j<(int)res.n; j++) {
                res.data[i*res.n + j] = mat->data[j];}}}
    return res;
}

Matrix2D MatrixApply(Matrix2D *mat, double(*pf)(double)) {
    Matrix2D res = MatrixMake(mat->m, mat->n);
    for(int i=0; i<(int)res.m; i++) {
        for(int j=0; j<(int)res.n; j++) {
            res.data[i*res.n + j] = pf(mat->data[i*mat->n + j]);}}
    return res;
}

Matrix2D MatrixSoftmax(Matrix2D *mat) {
    return MatrixMake(1,1);
}

Matrix2D MatrixAddScalar(Matrix2D *mat, double scalar) {
    Matrix2D res = MatrixMake(mat->m, mat->n);
    for(int i=0; i<(int)mat->m; i++){
        for(int j=0; j<(int)mat->n; j++){
            res.data[i*res.n + j] = mat->data[i*mat->n + j] + scalar;}}
    return res;
}

Matrix2D MatrixSubScalar(Matrix2D *mat, double scalar) {
    Matrix2D res = MatrixMake(mat->m, mat->n);
    for(int i=0; i<(int)mat->m; i++){
        for(int j=0; j<(int)mat->n; j++){
            res.data[i*res.n + j] = mat->data[i*mat->n + j] - scalar;}}
    return res;
}

Matrix2D MatrixMulScalar(Matrix2D *mat, double scalar) {
    Matrix2D res = MatrixMake(mat->m, mat->n);
    for(int i=0; i<(int)mat->m; i++){
        for(int j=0; j<(int)mat->n; j++){
            res.data[i*res.n + j] = mat->data[i*mat->n + j] * scalar;}}
    return res;
}

Matrix2D MatrixDivScalar(Matrix2D *mat, double scalar) {
    return MatrixMulScalar(mat, 1/scalar);
}

Matrix2D MatrixAddMatrix(Matrix2D *mat1, Matrix2D *mat2) {
    Matrix2D res;
    if(mat1->m == mat2->m && mat1->n == mat2->n) {
        res = MatrixMake(mat1->m, mat1->n);
        for(int i=0; i<(int)mat1->m; i++){
            for(int j=0; j<(int)mat1->n; j++){
                res.data[i*res.n + j] = mat1->data[i*mat1->n + j] +
                                        mat2->data[i*mat2->n + j];}}}
    else {
        if(mat1->m == mat2->m && mat2->n == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->n);
            res = MatrixAddMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}
        if(mat1->n == mat2->n && mat2->m == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->m);
            res = MatrixAddMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}}
    
    return res;
}

Matrix2D MatrixSubMatrix(Matrix2D *mat1, Matrix2D *mat2) {
    Matrix2D res;
    if(mat1->m == mat2->m && mat1->n == mat2->n) {
        res = MatrixMake(mat1->m, mat1->n);
        for(int i=0; i<(int)mat1->m; i++){
            for(int j=0; j<(int)mat1->n; j++){
                res.data[i*res.n + j] = mat1->data[i*mat1->n + j] -
                                        mat2->data[i*mat2->n + j];}}}
    else {
        if(mat1->m == mat2->m && mat2->n == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->n);
            res = MatrixSubMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}
        if(mat1->n == mat2->n && mat2->m == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->m);
            res = MatrixSubMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}}
    
    return res;
}

Matrix2D MatrixMulMatrix(Matrix2D *mat1, Matrix2D *mat2) {
    Matrix2D res;
    if(mat1->m == mat2->m && mat1->n == mat2->n) {
        res = MatrixMake(mat1->m, mat1->n);
        for(int i=0; i<(int)mat1->m; i++){
            for(int j=0; j<(int)mat1->n; j++){
                res.data[i*res.n + j] = mat1->data[i*mat1->n + j] *
                                        mat2->data[i*mat2->n + j];}}}
    else {
        if(mat1->m == mat2->m && mat2->n == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->n);
            res = MatrixMulMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}
        if(mat1->n == mat2->n && mat2->m == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->m);
            res = MatrixMulMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}}
    
    return res;
}

Matrix2D MatrixDivMatrix(Matrix2D *mat1, Matrix2D *mat2) {
    Matrix2D res;
    if(mat1->m == mat2->m && mat1->n == mat2->n) {
        res = MatrixMake(mat1->m, mat1->n);
        for(int i=0; i<(int)mat1->m; i++){
            for(int j=0; j<(int)mat1->n; j++){
                res.data[i*res.n + j] = mat1->data[i*mat1->n + j] /
                                        mat2->data[i*mat2->n + j];}}}
    else {
        if(mat1->m == mat2->m && mat2->n == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->n);
            res = MatrixDivMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}
        if(mat1->n == mat2->n && mat2->m == 1) {
            Matrix2D mat2_expanded = MatrixExpand(mat2, mat1->m);
            res = MatrixDivMatrix(mat1, &mat2_expanded);
            MatrixFree(&mat2_expanded);}}
    
    return res;
}

Matrix2D MatrixRead(char *filename, size_t m, size_t n) {
    FILE *fptr;

    fptr = fopen(filename, "r");
    if(fptr == NULL){
        printf("Cannot read Matrix file does not exists");
        exit(EXIT_FAILURE);
    }

    Matrix2D mat = MatrixMake(m, n);
    for(int i=0; i<(int)m; i++){
        for(int j=0; j<(int)n; j++){
            fscanf(fptr, "%lf", &mat.data[i*n + j]);}}

    return mat;
}

Matrix2D MatrixRandom(size_t m, size_t n) {

    srand(time(NULL));

    Matrix2D mat = MatrixMake(m, n);
    for(int i=0; i<(int)m; i++){
        for(int j=0; j<(int)n; j++){
            mat.data[i*n + j] = (double)rand() / (double)RAND_MAX - 0.5;}}

    return mat;
}
