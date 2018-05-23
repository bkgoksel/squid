# Download data to PROJECT_ROOT/data for local runs

# Connect to stanford codalab
ssh -N -f -n -L 12800:localhost:2800 -L 18000:localhost:8000 kerem@codalab.stanford.edu
cl alias nlp http://localhost:12800

cl work nlp::bkgoksel-neg
cl down train-v1.1.json
cl down dev-v1.1.json
cl down tiny-dev.json
cl down train-rule-unans-weighted.json
cl down dev-rule-unans.json
cl down train-v2.0.json
cl down dev-v2.0.json
cl down generate-tfidf-train-2
cl down squad_dev_tfidf_n2.json
cl down train-nosent-neg.json
cl down dev-nosent-neg.json

mkdir data/original
mv train-v1.1.json original/
mv dev-v1.1.json original/

mkdir data/negatives

mkdir data/negatives/rule
mv train-rule-unans-weighted.json data/negatives/rule/
mv dev-rule-unans.json data/negatives/rule/

mkdir data/negatives/squadrun
mv train-v2.0.json data/negatives/squadrun/
mv dev-v2.0.json data/negatives/squadrun/

mkdir data/negatives/tfidf
mv generate-tfidf-train-2 data/negatives/tfidf/
mv squad_dev_tfidf_n2.json data/negatives/tfidf/

mkdir data/negatives/nosent
mv train-nosent-neg.json data/negatives/nosent/
mv dev-nosent-neg.json data/negatives/nosent/
