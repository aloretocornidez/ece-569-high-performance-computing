cmake ../build_dir/ ; make -j; ./BasicMatrixMultiplication_DatasetGenerator; for test in {0..9}; do(echo "test: $test"; ./BasicMatrixMultiplication_Solution -e MatrixMultiplication/Dataset/${test}/output.raw -i MatrixMultiplication/Dataset/${test}/input0.raw,MatrixMultiplication/Dataset/${test}/input1.raw -o my-output-${test}.raw -t matrix | grep "correctq"; echo "";); done;