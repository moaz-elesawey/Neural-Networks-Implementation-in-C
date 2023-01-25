# set terminal qt
set terminal pngcairo
set output "plot.png"
set multiplot layout 2, 1

set xlabel "Epoch"
set ylabel "Loss"
set title "XOR Classification"
plot 'loss.xor.dat' using 1 title "" with lines

set xlabel "Epoch"
set ylabel "Loss"
set title "Sklearn Blobs Classification"
plot 'loss.blobs.dat' using 1 title "" with lines
