for i in `seq 0 20`:
do
	echo $i
	raspistill -vf -w 600 -h 600 -roi 0.46,0.34,0.25,0.25 -o test/right${i}.jpg
done
