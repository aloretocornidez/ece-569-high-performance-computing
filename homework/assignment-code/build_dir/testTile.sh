cmake ../build_dir/

make -j

for iteration in {0..9}; do
       for test in {0..3}; do
              (
                     echo "test: $test iteration: $iteration"
                     ./TiledMatrixMultiplication_Solution -e TiledMatrixMultiplication/Dataset/${test}/output.raw -i TiledMatrixMultiplication/Dataset/${test}/input0.raw,TiledMatrixMultiplication/Dataset/${test}/input1.raw -o my-output-${test}.raw -t matrix >> "test-${test}.txt"
                     echo ""
              )
       done
done
