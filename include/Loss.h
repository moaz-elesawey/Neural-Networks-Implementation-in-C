#ifndef _LOSS_H_
#define _LOSS_H_

#include "Matrix2D.h"
#include <stdio.h>


double MSELoss(Matrix2D *y, Matrix2D *h) {
    Matrix2D t1 = MatrixSubMatrix(y, h);
    Matrix2D t2 = MatrixMulMatrix(&t1, &t1);
    Matrix2D t3 = MatrixSum(&t2, 1);

    double J = t3.data[0] / y->m;

    MatrixFree(&t1); MatrixFree(&t2); MatrixFree(&t3); 
    
    return J;
}

#endif /* _LOSS_H_ */
