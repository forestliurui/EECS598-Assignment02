rm cpu
rm mem
$1 &
pid=$!
echo $pid
while true; do
    ./calcCPU.sh $pid >> cpu &
    ./calcMem.sh $pid >> mem &
    sleep 2
done

