rm cpu
rm mem
$1 &
pid=$2
echo $pid
while true; do
    ./calcCPU.sh $pid >> cpu &
    ./calcMem.sh $pid >> mem &
    sleep 2
done

