cd result_22-8/eps-scale/
rm diff_1000_100 pot_1000_100 ham_1000_100 diff-k_1000_100 kin_1000_100 ratio_1000_100 ratio-k_1000_100
cd ../../
for i in 0.1 0.01 0.001 0.0001 0.00001
do
    python ann-fixed.py 1000 100 $i > result_22-8/eps-scale/out_1000_100_$i
    grep "diff-h" result_22-8/eps-scale/out_1000_100_$i | head -n 1 | cut -d":" -f2 > result_22-8/eps-scale/diff_1000_100_$i
    grep "diff-k" result_22-8/eps-scale/out_1000_100_$i | head -n 1 | cut -d":" -f2 > result_22-8/eps-scale/diff-k_1000_100_$i
    grep "current U" result_22-8/eps-scale/out_1000_100_$i | head -n 1 | cut -d":" -f2 > result_22-8/eps-scale/pot_1000_100_$i
    grep "current H" result_22-8/eps-scale/out_1000_100_$i | head -n 1 | cut -d":" -f2 > result_22-8/eps-scale/ham_1000_100_$i
    grep "current K" result_22-8/eps-scale/out_1000_100_$i | head -n 1 | cut -d":" -f2 > result_22-8/eps-scale/kin_1000_100_$i
    grep "ratio-h" result_22-8/eps-scale/out_1000_100_$i | head -n 1 | cut -d":" -f2 > result_22-8/eps-scale/ratio_1000_100_$i
    grep "ratio-k" result_22-8/eps-scale/out_1000_100_$i | head -n 1 | cut -d":" -f2 > result_22-8/eps-scale/ratio-k_1000_100_$i

    cat result_22-8/eps-scale/diff_1000_100_$i >> result_22-8/eps-scale/diff_1000_100
    cat result_22-8/eps-scale/pot_1000_100_$i >> result_22-8/eps-scale/pot_1000_100
    cat result_22-8/eps-scale/ham_1000_100_$i >> result_22-8/eps-scale/ham_1000_100
    cat result_22-8/eps-scale/diff-k_1000_100_$i >> result_22-8/eps-scale/diff-k_1000_100
    cat result_22-8/eps-scale/kin_1000_100_$i >> result_22-8/eps-scale/kin_1000_100
    cat result_22-8/eps-scale/ratio_1000_100_$i >> result_22-8/eps-scale/ratio_1000_100
    cat result_22-8/eps-scale/ratio-k_1000_100_$i >> result_22-8/eps-scale/ratio-k_1000_100
done

python plot-eps-scale.py
