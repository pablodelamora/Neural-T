# Script for showing ratios of wins of a NN and random in the Conecta-T game vs total games.
# Call ./analyse.sh NN-File

echo "Competing..."
> tmp
for i in {1..100}; do
  python3 ../Neural-T/judge_alt.py $1 |
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
rm tmp
