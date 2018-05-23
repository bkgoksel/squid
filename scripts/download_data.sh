# Download data to PROJECT_ROOT/data for local runs

# Connect to stanford codalab
ssh -N -f -n -L 12800:localhost:2800 -L 18000:localhost:8000 kerem@codalab.stanford.edu
cl alias nlp http://localhost:12800

cl work nlp::bkgoksel-neg
cl down train-v1.1.json
cl down dev-v1.1.json
cl down tiny-dev.json
cl down train-rule-unans-weighted.json
l down dev-rule-unans.json
cl down train-v2.0.json
cl down dev-v2.0.json
cl down generate-tfidf-train-2
cl down squad_dev_tfidf_n2.json
cl down train-nosent-neg.json
cl down dev-nosent-neg.json

mkdir data/original
mv train-v1.1.json data/original/train.json
mv dev-v1.1.json data/original/dev.json
mv tiny-dev.json data/original/tiny-dev.json

mkdir data/negatives

mkdir data/negatives/rule
mv train-rule-unans-weighted.json data/negatives/rule/train.json
mv dev-rule-unans.json data/negatives/rule/dev.json

mkdir data/negatives/squadrun
mv train-v2.0.json data/negatives/squadrun/train.json
mv dev-v2.0.json data/negatives/squadrun/dev.json

mkdir data/negatives/tfidf
mv generate-tfidf-train-2/tfidf.json data/negatives/tfidf/train.json
mv squad_dev_tfidf_n2.json data/negatives/tfidf/dev.json
rm -rf generate-tfidf-train-2

mkdir data/negatives/nosent
mv train-nosent-neg.json data/negatives/nosent/train.json
mv dev-nosent-neg.json data/negatives/nosent/dev.json
