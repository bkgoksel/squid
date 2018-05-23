# Work plan:

- [ ] Prepare harness for tokenization, batch building and evaluation
- [ ] Make a basic LSTM->Dense->Spans&no-answer outputting model to get the whole training/testing process running
- [ ] Implement BiDAF on top
- [ ] Integrate ELMo vectors
- [ ] Think about data cleanup, tokenization and all the other shenanigans of working with SQuAD
- [ ] GAN setup for negative generation


## Q-A Representation
* Have some objects to represent the questions textually?
* Have some encoder that takes these textual objects and encodes them numerically
* Saveability to disk/reloadability
