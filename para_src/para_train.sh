echo "number of GPUs:\c"
read num
echo "number of GO terms:\c"
read n
sh train_new_model.sh 1 $n $num
