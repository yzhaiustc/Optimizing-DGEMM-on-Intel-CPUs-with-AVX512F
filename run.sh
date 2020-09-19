rm res*
for i in {100..2000..100}
do
		export MKL_NUM_THREADS=1
		export OMP_NUM_THREADS=1
		./k1 $i $i $i >> res_1.txt
		./k2 $i $i $i >> res_2.txt
		./k3 $i $i $i >> res_3.txt
		./k4 $i $i $i >> res_4.txt
		./k5 $i $i $i >> res_5.txt
		./k6 $i $i $i >> res_6.txt
		./k7 $i $i $i >> res_7.txt
		./k8 $i $i $i >> res_8.txt
		./k9 $i $i $i >> res_9.txt
done
