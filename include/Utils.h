#ifndef _UTILS_H_
#define _Utils_H_

#include "Matrix2D.h"


double ReLU(double z){
    return z >= 0 ? z : 0;
}
double dReLU(double z){
    return z >= 0 ? 1 : 0;
}

double Sigmoid(double z) {
    return 1. / (1. + exp(-z));
}

double dSigmoid(double z) {
    double A = Sigmoid(z);
    return A * (1.0 - A);
}

/**
 * Computes the Accuracy Score betwenn two matriceis
 */
double AccuracyScore(Matrix2D *mat1, Matrix2D *mat2) {
    int true_counts = 0;
    for(int i=0; i<(int)mat1->m; i++) {
        if((int)round(mat2->data[i]) == (int)mat1->data[i])
            true_counts++;}
    return (double)true_counts / (double)(mat1->m);
}

/**
 * Compute the R2 Score between two matrecies y_true and y_pred
 */
double R2Score(Matrix2D *y_true, Matrix2D *y_pred) {
    Matrix2D diff = MatrixSubMatrix(y_true, y_pred);
    Matrix2D diff_sq = MatrixPow(&diff, 2);
    Matrix2D SS_res = MatrixSum(&diff_sq, 1);
    double SS_res_value = SS_res.data[0];

    Matrix2D T1 = MatrixSum(y_true, 1);
    Matrix2D T2 = MatrixDivScalar(&T1, y_true->n);
    double y_mu = T2.data[0];

    Matrix2D diff_2 = MatrixSubScalar(y_true, y_mu);
    Matrix2D diff_2_sq = MatrixPow(&diff_2, 2);
    Matrix2D SS_tot = MatrixSum(&diff_2_sq, 1);
    double SS_tot_value = SS_tot.data[0];

    double R2 = 1 - (SS_res_value / SS_tot_value);

    MatrixFree(&diff); MatrixFree(&diff_sq);
    MatrixFree(&diff_2); MatrixFree(&diff_2_sq);
    MatrixFree(&T1); MatrixFree(&T2);
    MatrixFree(&SS_res); MatrixFree(&SS_tot);

    return R2;
}

#endif
