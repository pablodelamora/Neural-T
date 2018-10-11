# gtc for Generate (samples), Train (neural network) and Compete (Conecta-T)
function clean() {
  rm *.samples
  rm *.nn.json
  rm -r __pycache__
  rm tmp
}

echo "Generating samples..."
SAMPLES=$(./gensamples.sh 100)

echo "Training neural network..."
NN=$(python3 neural_network.py --scripting -r 0.00001 $SAMPLES) # Try running <python3 neural_network.py -h> for more options

echo "Competing..."
> tmp
for i in {1..100}; do
  python3 judge_alt.py $NN |
  tail -n 1 >> tmp
done

# Strip tmp file of empty lines produced by errors of judge_alt.py
sed '/^$/d' tmp > tmp2
mv tmp2 tmp

# Report findings
echo "NN wins:"
grep "The winner is  2" tmp | wc -l
echo "Random Generator wins:"
grep -v "The winner is  2" tmp | wc -l
echo "Total games:"
cat tmp | wc -l

# Remove garbage
clean
