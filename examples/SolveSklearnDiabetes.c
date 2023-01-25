#include "Matrix2D.h"
#include "Loss.h"
#include "Utils.h"

#define N_EPOCHS 500


void SolveSklearnBlobs() {
    size_t m = 442, n = 10;
    size_t d = 32, k = 1;
    double alpha = 0.01;

    Matrix2D X_data = MatrixRead("data/diabetes.x.dat", m, n);
    Matrix2D X = MatrixT(&X_data);
    MatrixFree(&X_data);

    Matrix2D Y_data = MatrixRead("data/diabetes.y.dat", m, 1);
    Matrix2D Y = MatrixT(&Y_data);
    MatrixFree(&Y_data);

    Matrix2D W1 = MatrixRandom(d , n);
    Matrix2D W2 = MatrixRandom(k , d);

    Matrix2D b1 = MatrixRandom(d, 1);
    Matrix2D b2 = MatrixRandom(k, 1);

    MatrixShow(&Y);

    for(int i=0; i<N_EPOCHS; i++){

        /* Forward Propagation */
        Matrix2D T1 = MatrixDot(&W1, &X); /* Temporary Matrix */
        Matrix2D Z1 = MatrixAddMatrix(&T1, &b1);
        Matrix2D A1 = MatrixApply(&Z1, Sigmoid);

        Matrix2D T2 = MatrixDot(&W2, &A1); /* Temporary Matrix */
        Matrix2D Z2 = MatrixAddMatrix(&T2, &b2);
        Matrix2D A2 = MatrixApply(&Z2, Sigmoid);

        /* Performance Measure */
        double J = MSELoss(&A2, &Y);
        double acc = AccuracyScore(&Y, &A2);
        double R2 = R2Score(&Y, &A2);
        printf("Epoch [%4d-%d], Loss = %10.5f, Accuracy = %10.5f, R2 Score = %10.4f\n",
                i, N_EPOCHS, J, acc, R2);

        /* Backward Propagation */
        /****** Second Layer ******/
        Matrix2D dZ2 = MatrixSubMatrix(&A2, &Y);
        Matrix2D T3  = MatrixT(&A1);
        Matrix2D T4  = MatrixDot(&dZ2, &T3);
        Matrix2D dW2 = MatrixMulScalar(&T4, alpha/m);
        Matrix2D T5  = MatrixSum(&dZ2, 1);
        Matrix2D db2 = MatrixMulScalar(&T5, alpha/m);

        /* Update the Gradients */
        Matrix2D T6  = MatrixSubMatrix(&W2, &dW2);
        Matrix2D T7  = MatrixSubMatrix(&b2, &db2);
        MatrixCopy(&W2, &T6);
        MatrixCopy(&b2, &T7);

        /* Print To Console */
        /* MatrixShow(&W2); MatrixShow(&b2); */
        /***************************/

        /******** First Layer *********/
        Matrix2D T8  = MatrixT(&W2);
        Matrix2D T9  = MatrixDot(&T8, &dZ2);
        Matrix2D T10 = MatrixApply(&Z1, dSigmoid);
        Matrix2D dZ1 = MatrixMulMatrix(&T9, &T10);
        Matrix2D T11 = MatrixT(&X);
        Matrix2D T12 = MatrixDot(&dZ1, &T11);
        Matrix2D dW1 = MatrixMulScalar(&T12, alpha/m);
        Matrix2D T13 = MatrixSum(&dZ1, 1);
        Matrix2D db1 = MatrixMulScalar(&T13, alpha/m);

        /* Update the Gradients */
        Matrix2D T14 = MatrixSubMatrix(&W1, &dW1);
        Matrix2D T15 = MatrixSubMatrix(&b1, &db1);
        MatrixCopy(&W1, &T14);
        MatrixCopy(&b1, &T15);

        /* Print To Console */
        /******************************/
        /* MatrixShow(&W1); MatrixShow(&b1); */

        /* if(i == N_EPOCHS-1) { */
        /*  MatrixShow(&A2); */
        /* } */
        
        /* Free The Memory */
        MatrixFree(&T1);  MatrixFree(&T2);
        MatrixFree(&T3);  MatrixFree(&T4);
        MatrixFree(&T5);  MatrixFree(&T6);
        MatrixFree(&T7);  MatrixFree(&T8);
        MatrixFree(&T9);  MatrixFree(&T10);
        MatrixFree(&T11); MatrixFree(&T12);
        MatrixFree(&T13); MatrixFree(&T14);

        MatrixFree(&Z1);  MatrixFree(&Z2);
        MatrixFree(&A1);  MatrixFree(&A2);

        MatrixFree(&dZ1); MatrixFree(&dZ2);
        MatrixFree(&dW1); MatrixFree(&db1);
        MatrixFree(&dW2); MatrixFree(&db2);
    }
    printf("\nPress any key to exit..."); getchar();

    MatrixFree(&X);
    MatrixFree(&Y);

    MatrixFree(&W1);
    MatrixFree(&W2);

    MatrixFree(&b1);
    MatrixFree(&b2);
}

int main(int argc, char **argv) {

    SolveSklearnBlobs();

    return EXIT_SUCCESS;
}

