while true; do
ps -C <ProgramName> -o pid=,%mem=,vsz= >> /tmp/mem.log
gnuplot /tmp/gnuplot.script
sleep 1
done &
