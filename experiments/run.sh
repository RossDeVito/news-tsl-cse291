echo "Running crisis orig"
python evaluate.py --dataset ../datasets/crisis --method datewise --resources ../resources/datewise --output ../results/crisis.clust.json --model orig > crisis.full_eval.orig.txt
echo "Running crisis new"
python evaluate.py --dataset ../datasets/crisis --method datewise --resources ../resources/datewise --output ../results/crisis.clust.json --model new_lr > crisis.full_eval.new_lr.txt

echo "Running entities orig"
python evaluate.py --dataset ../datasets/entities --method datewise --resources ../resources/datewise --output ../results/entities.clust.json --model orig > entities.full_eval.orig.txt
echo "Running entities new"
python evaluate.py --dataset ../datasets/entities --method datewise --resources ../resources/datewise --output ../results/entities.clust.json --model new_lr > entities.full_eval.new_lr.txt