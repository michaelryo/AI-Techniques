cd ./Negative_Words
copy *.txt NegativeWordsCompile.txt
move  NegativeWordsCompile.txt  ../

cd ../Positive_Words
copy *.txt PositiveWordsCompile.txt
move  PositiveWordsCompile.txt ../



@echo off
echo Finish
set /p id="Please press Enter to Close"