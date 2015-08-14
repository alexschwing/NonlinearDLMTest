#!/bin/bash
parallel -j16 ../NonlinearTest {1} {2} 1 2 '>' 10000svm-{1}-{2}.txt ::: 1 0.3 0.1 0.03 0.01 0.003 0.001 ::: 1 0.1 0.01 0.001
