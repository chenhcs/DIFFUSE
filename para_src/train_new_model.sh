#!/bin
for i in $(seq 0 $(( $3 - 1)))
do
{
	for j in $(seq $1 $3 $2)
	do
		if [ $(($j+$i)) -lt $(($2+1)) ]
		then
			echo "train model $(($j+$i))"
			python model.py $(($j+$i)) $i 2>/dev/null
		fi
	done
}&
done
wait
echo "all the models are finished"
