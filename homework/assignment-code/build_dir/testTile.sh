cmake ../build_dir/

make -j

rm -rf ./MatrixMultiplication/*
rm -rf ./TiledMatrixMultiplication/*

./TiledMatrixMultiplication_DatasetGenerator
./BasicMatrixMultiplication_DatasetGenerator



for iteration in {0..9}; do
    for test in {0..9}; do
        (
            echo "Tile test: $test iteration: $iteration"
            echo "test: $test iteration: $iteration" >>"tiled-test-${test}.txt"
            ./TiledMatrixMultiplication_Solution -e TiledMatrixMultiplication/Dataset/${test}/output.raw -i TiledMatrixMultiplication/Dataset/${test}/input0.raw,TiledMatrixMultiplication/Dataset/${test}/input1.raw -o my-tiled-output-${test}.raw -t matrix >>"tiled-test-${test}.txt"
            echo "" >>"tiled-test-${test}.txt"
            echo ""

            echo "Basic test: $test iteration: $iteration"
            echo "test: $test iteration: $iteration" >>"basic-test-${test}.txt"
            ./BasicMatrixMultiplication_Solution -e MatrixMultiplication/Dataset/${test}/output.raw -i MatrixMultiplication/Dataset/${test}/input0.raw,MatrixMultiplication/Dataset/${test}/input1.raw -o my-basic-output-${test}.raw -t matrix >>"basic-test-${test}.txt"
            echo "" >>"basic-test-${test}.txt"
            echo ""
        )
    done
done
