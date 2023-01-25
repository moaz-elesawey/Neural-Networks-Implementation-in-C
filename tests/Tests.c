#include "Matrix2D.h"
#include "Loss.h"


double ReLU(double z){
	return z >= 0 ? z : 0;
}
double Sigmoid(double z) {
	return 1. / (1. + exp(-z));
}

void TestMatrixT() {
	Matrix2D mat = MatrixMake(3, 2);
	mat.data[0] = 1; mat.data[1] = 2;
	mat.data[2] = 3; mat.data[3] = 4;
	mat.data[4] = 5; mat.data[5] = 6;

	Matrix2D matT = MatrixT(&mat);

	MatrixShow(&mat);
	MatrixShow(&matT);

	MatrixFree(&mat);
	MatrixFree(&matT);
}

void TestMatrixDot() {
	Matrix2D a, b, c;

	a = MatrixMake(3, 2);
	a.data[0] = 1; a.data[1] = 2;
	a.data[2] = 3; a.data[3] = 4;
	a.data[4] = 5; a.data[5] = 6;
	
	b = MatrixMake(2, 1);
	b.data[0] = 1; b.data[1] = 2;

	c = MatrixDot(&a, &b);

	MatrixShow(&a);
	MatrixShow(&b);
	MatrixShow(&c);

	MatrixFree(&a);
	MatrixFree(&b);
	MatrixFree(&c);
}

void TestMatrixSum() {
	Matrix2D a = MatrixMake(3, 2);
	a.data[0] = 1; a.data[1] = 2;
	a.data[2] = 3; a.data[3] = 4;
	a.data[4] = 5; a.data[5] = 6;

	Matrix2D b = MatrixSum(&a, 0);
	Matrix2D c = MatrixSum(&a, 1);

	Matrix2D d = MatrixSum(&b, 1);

	MatrixShow(&a);
	MatrixShow(&b);
	MatrixShow(&c);
	MatrixShow(&d);

	MatrixFree(&a);
	MatrixFree(&b);
	MatrixFree(&c);
	MatrixFree(&d);
}

void TestMatrixExpand() {
	Matrix2D a, b, c, d;

	a = MatrixMake(3, 1);
	a.data[0] = 1; a.data[1] = 2;
	a.data[2] = 3;

	b = MatrixMake(1, 2);
	b.data[0] = 1; b.data[1] = 2;

	c = MatrixExpand(&a, 2);
	MatrixShow(&a); MatrixShow(&c);

	d = MatrixExpand(&b, 3);
	MatrixShow(&b); MatrixShow(&d);

	MatrixFree(&a); MatrixFree(&b);
	MatrixFree(&c); MatrixFree(&d);
}

void TestMatrixElementWizeOperations() {
	Matrix2D a, b, c, d, e;

	a = MatrixMake(3, 2);
	a.data[0] = 1; a.data[1] = 2;
	a.data[2] = 3; a.data[3] = 4;
	a.data[4] = 5; a.data[5] = 6;

	b = MatrixAddScalar(&a, 1.0);
	c = MatrixSubScalar(&a, 1.0);
	d = MatrixMulScalar(&a, 2.0);
	e = MatrixDivScalar(&a, 3.0);

	MatrixShow(&a); MatrixShow(&b);
	MatrixShow(&c); MatrixShow(&d);
	MatrixShow(&e);

	MatrixFree(&a); MatrixFree(&b);
	MatrixFree(&c); MatrixFree(&d);
	MatrixFree(&e);
}

void TestMatrixMatrixOperations() {
	Matrix2D a, b, c, d, e;

	a = MatrixMake(3, 2);
	a.data[0] = 1; a.data[1] = 2;
	a.data[2] = 3; a.data[3] = 4;
	a.data[4] = 5; a.data[5] = 6;

	b = MatrixMake(3, 2);
	b.data[0] = 2; b.data[1] = 2;
	b.data[2] = 2; b.data[3] = 2;
	b.data[4] = 2; b.data[5] = 2;

	c = MatrixMake(3, 1);
	c.data[0] = 2; c.data[1] = 2;
	c.data[2] = 2;

	d = MatrixMake(1, 2);
	d.data[0] = 2; d.data[1] = 2;

	e = MatrixAddMatrix(&a, &d);
	MatrixShow(&e);
	MatrixFree(&e);
	
	e = MatrixSubMatrix(&a, &d);
	MatrixShow(&e);
	MatrixFree(&e);
	
	e = MatrixMulMatrix(&a, &d);
	MatrixShow(&e);
	MatrixFree(&e);

	e = MatrixDivMatrix(&a, &d);
	MatrixShow(&e);
	MatrixFree(&e);

	MatrixFree(&a); MatrixFree(&b);
	MatrixFree(&c); MatrixFree(&d);

}

void TestMatrixApply() {
	Matrix2D a, b;
	a = MatrixMake(3, 2);
	a.data[0] = -1; a.data[1] = 2;
	a.data[2] = -1; a.data[3] = 2;
	a.data[4] = -1; a.data[5] = 2;

	b = MatrixApply(&a, ReLU);
	MatrixShow(&b);

	b = MatrixApply(&a, Sigmoid);
	MatrixShow(&b);

	MatrixFree(&a); MatrixFree(&b);
}

void TestMatrixRead() {
	Matrix2D mat = MatrixRead("test_mat.txt", 3, 2);
	MatrixShow(&mat);
	MatrixFree(&mat);
}

void TestMatrixRandom() {
	Matrix2D mat = MatrixRandom(4, 4);
	MatrixShow(&mat);
	MatrixFree(&mat);
}

void TestMSELoss() {
	Matrix2D h = MatrixMake(1, 4);
	h.data[0] = 1; h.data[1] = 0;
	h.data[2] = 0; h.data[3] = 1;

	Matrix2D y = MatrixMake(1, 4);
	y.data[0] = 1; y.data[1] = 1;
	y.data[2] = 0; y.data[3] = 1;

	double J = MSELoss(&y, &h);
	printf("J(y, h) = %10.5lf\n", J);

	MatrixFree(&h); MatrixFree(&y);
}

int main(int argc, char **argv) {

	/* TestMatrixT(); */
	/* TestMatrixDot(); */
	/* TestMatrixSum(); */
	/* TestMatrixExpand(); */
	/* TestMatrixElementWizeOperations(); */
	/* TestMatrixMatrixOperations(); */
	/* TestMatrixApply(); */
	/* TestMatrixRead(); */
	/* TestMatrixRandom(); */
	TestMSELoss();

	return EXIT_SUCCESS;
}

