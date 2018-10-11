FILE_NAME=$(date "+%y%m%d%H%M%S").samples
echo "[" > $FILE_NAME;
for i in $(seq 1 $1); do
  python3 gensamples_phase1.py |
  python3 gensamples_phase2.py >> $FILE_NAME;
done
echo "]" >> $FILE_NAME;
echo $FILE_NAME
