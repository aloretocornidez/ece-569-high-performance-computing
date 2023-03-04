cmake ../build_dir/
make -j

for iteration in {0..9}; do
    for test in {0..3}; do
        (
            echo "test: $test iteration: $iteration"
            echo "test: $test iteration: $iteration" >>"test-${test}.txt"
            ./BasicMatrixMultiplication_Solution -e MatrixMultiplication/Dataset/${test}/output.raw -i MatrixMultiplication/Dataset/${test}/input0.raw,MatrixMultiplication/Dataset/${test}/input1.raw -o my-output-${test}.raw -t matrix >>"test-${test}.txt"
            echo "" >>"test-${test}.txt"
            echo ""
        )
    done
done
