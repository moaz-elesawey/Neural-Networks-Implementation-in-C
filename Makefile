
all: xor blobs diabetes

xor:
	cc examples/SolveXOR.c src/Matrix2D.c -Iinclude -lm -o bin/xor.out

blobs:
	cc examples/SolveSklearnBlobs.c src/Matrix2D.c -Iinclude -lm -o bin/blobs.out

diabetes:
	cc examples/SolveSklearnDiabetes.c src/Matrix2D.c -Iinclude -lm -o bin/diabetes.out

test:
	cc tests/Tests.c src/Matrix2D.c -Iinclude -lm -o bin/test.out

clean:
	rm bin/*
