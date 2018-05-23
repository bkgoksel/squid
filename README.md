# Work plan:

- [ ] Prepare harness for tokenization, batch building and evaluation
- [ ] Make a basic LSTM->Dense->Spans&no-answer outputting model to get the whole training/testing process running
- [ ] Implement BiDAF on top
- [ ] Integrate ELMo vectors
- [ ] Think about data cleanup, tokenization and all the other shenanigans of working with SQuAD
- [ ] GAN setup for negative generation


## Q-A Representation
* Have some objects to represent the questions textually [x]
* Have some encoder that takes these textual objects and encodes them numerically [x]
  * List[token_idx] ?
  * List[List[token_idx]] ?
* Saveability to disk/reloadability [x]
* Random batching? <Context, Question, Answers> [x]
* Tokenization [ ]

## Model Architecture

* Predictor model: encoded sample -> ((span_start, span_end), no_ans)
* Evaluator model: ((span_start, span_end), no_ans) -> loss
