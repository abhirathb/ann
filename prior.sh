#!/bin/bash
prefix=result_22-8/prior-cont
cd $prefix
#rm diff_1000_100 pot_1000_100 ham_1000_100 diff-k_1000_100 kin_1000_100 ratio_1000_100 ratio-k_1000_100
rm  eps-scale/*
rm var-scale/*
cd ../../


##############
echo "Starting the epsilon-scale study"
sub=eps-scale

for i in 0.1 0.01 0.001 0.0001 0.00001
do
    python ann-prior-fixed.py 1000 100 $i > $prefix/$sub/out_1000_100_$i
    grep "diff-h" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/diff_1000_100_$i
    grep "diff-k" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/diff-k_1000_100_$i
    grep "diff-u" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/diff-u_1000_100_$i
    grep "diff-l" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/diff-l_1000_100_$i
    grep "current U" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/pot_1000_100_$i
    grep "current H" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/ham_1000_100_$i
    grep "current K" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/kin_1000_100_$i
    grep "current L" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/log_1000_100_$i
    grep "ratio-h" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/ratio_1000_100_$i
    grep "ratio-k" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/ratio-k_1000_100_$i
    grep "ratio-u" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/ratio-u_1000_100_$i
    grep "ratio-l" $prefix/$sub/out_1000_100_$i | head -n 1 | cut -d":" -f2 > $prefix/$sub/ratio-l_1000_100_$i

    cat $prefix/$sub/diff_1000_100_$i >> $prefix/$sub/diff_1000_100
    cat $prefix/$sub/pot_1000_100_$i >> $prefix/$sub/pot_1000_100
    cat $prefix/$sub/ham_1000_100_$i >> $prefix/$sub/ham_1000_100
    cat $prefix/$sub/log_1000_100_$i >> $prefix/$sub/log_1000_100
    cat $prefix/$sub/diff-k_1000_100_$i >> $prefix/$sub/diff-k_1000_100
    cat $prefix/$sub/diff-u_1000_100_$i >> $prefix/$sub/diff-u_1000_100
    cat $prefix/$sub/diff-l_1000_100_$i >> $prefix/$sub/diff-l_1000_100
    cat $prefix/$sub/kin_1000_100_$i >> $prefix/$sub/kin_1000_100
    cat $prefix/$sub/ratio_1000_100_$i >> $prefix/$sub/ratio_1000_100
    cat $prefix/$sub/ratio-k_1000_100_$i >> $prefix/$sub/ratio-k_1000_100
    cat $prefix/$sub/ratio-u_1000_100_$i >> $prefix/$sub/ratio-u_1000_100
    cat $prefix/$sub/ratio-l_1000_100_$i >> $prefix/$sub/ratio-l_1000_100
done

python  plot-eps-scale-prior.py

echo "END of epsilon-scale study"
###############################


##############################
echo " Start of variance study"
sub=var-scale
prefix=$prefix/$sub
for i in 0.1 0.5 1.0 2.0 5.0 10.0 25.0 50.0 100.0
do
    for j in {100..10000..100}
    do
         suffix=$j"_100_0.00001_$i"
         out=$prefix/out_$suffix
         echo $out
         python ann-prior.py $j 100 0.00001 $i > $out
         grep "diff-h" $out | head -n 1 | cut -d":" -f2 > $prefix/diff_$suffix
         grep "diff-k" $out | head -n 1 | cut -d":" -f2 > $prefix/diff-k_$suffix
         grep "diff-u" $out | head -n 1 | cut -d":" -f2 > $prefix/diff-u_$suffix
         grep "diff-l" $out | head -n 1 | cut -d":" -f2 > $prefix/diff-l_$suffix
         grep "current U" $out | head -n 1 | cut -d":" -f2 > $prefix/pot_$suffix
         grep "current H" $out | head -n 1 | cut -d":" -f2 > $prefix/ham_$suffix
         grep "current K" $out | head -n 1 | cut -d":" -f2 > $prefix/kin_$suffix
         grep "current L" $out | head -n 1 | cut -d":" -f2 > $prefix/log_$suffix
         grep "ratio-h" $out | head -n 1 | cut -d":" -f2 > $prefix/ratio_$suffix
         grep "ratio-k" $out | head -n 1 | cut -d":" -f2 > $prefix/ratio-k_$suffix
         grep "ratio-u" $out | head -n 1 | cut -d":" -f2 > $prefix/ratio-u_$suffix
         grep "ratio-l" $out | head -n 1 | cut -d":" -f2 > $prefix/ratio-l_$suffix

         cat $prefix/diff_$suffix>> $prefix/diff_$i
         cat $prefix/pot_$suffix >> $prefix/pot_$i
         cat $prefix/ham_$suffix>> $prefix/ham_$i
         cat $prefix/log_$suffix>> $prefix/log_$i
         cat $prefix/diff-k_$suffix >> $prefix/diff-k_$i
         cat $prefix/diff-u_$suffix >> $prefix/diff-u_$i
         cat $prefix/diff-l_$suffix >> $prefix/diff-l_$i
         cat $prefix/kin_$suffix >> $prefix/kin_$i
         cat $prefix/ratio_$suffix >> $prefix/ratio_$i
         cat $prefix/ratio-k_$suffix >> $prefix/ratio-k_$i
         cat $prefix/ratio-u_$suffix >> $prefix/ratio-u_$i
         cat $prefix/ratio-l_$suffix >> $prefix/ratio-l_$i
     done
 done
python plot-var-scale-prior.py

echo " DONE "
