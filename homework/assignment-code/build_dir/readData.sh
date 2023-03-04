echo "Tiled Tests" >tiledData.txt
echo "Basic Tests" >basicData.txt
for i in {0..3}; do

    (
        echo "Case ${i}" >>tiledData.txt
        # cat tiled-test-${i}.txt | grep CUDA | awk '{print substr($3, 1, length($3)-1)}' >> tiledData.txt
        # cat tiled-test-${i}.txt | grep CUDA | awk '{print $3}' >>tiledData.txt
        cat tiled-test-${i}.txt | grep CUDA | awk '{print $3}' | sed 's/,*$//g' >> tiledData.txt

        echo "" >>tiledData.txt
        echo "Case ${i}" >>basicData.txt
        # cat basic-test-${i}.txt | grep CUDA | awk '{print substr($3, 1, length($3)-1)}' >> basicData.txt
        # cat basic-test-${i}.txt | grep CUDA | awk '{print $3}' >>basicData.txt
        cat basic-test-${i}.txt | grep CUDA | awk '{print $3}' | sed 's/,*$//g' >> basicData.txt

        echo "" >>basicData.txt
    
    )

done

