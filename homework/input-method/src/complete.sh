for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
	for j in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
	do
		python3 pinyin.py -i ./测试语料/input_2.txt -o ./test.txt -c $i $j
	done
done
for i in 0.9 0.99 0.999 0.9999 0.99999 1
do
        for j in 0.9 0.99 0.999 0.9999 0.99999 1
        do
                python3 pinyin.py -i ./测试语料/input_2.txt -o ./test.txt -c $i $j
        done
done
