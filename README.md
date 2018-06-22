# Work plan:

- [x] Prepare harness for tokenization, batch building and evaluation
- [x] Make a basic LSTM->Dense->Spans&no-answer outputting model to get the whole training/testing process running
- [x] Think about data cleanup, tokenization and all the other shenanigans of working with SQuAD
  - [ ] Lowercasing
  - [ ] Dealing with abbreviations
  - [ ] Dealing with numbers, dates etc
- [ ] Add unit testing for core components
- [ ] Make GPU compatible
- [ ] Debug simple architecture
- [ ] Implement BiDAF on top
- [ ] Integrate ELMo vectors
- [ ] Semi-supervised approach to make models resilient to single-word swaps
