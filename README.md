# Work plan:

- [x] Prepare harness for tokenization, batch building and evaluation
- [x] Make a basic LSTM->Dense->Spans&no-answer outputting model to get the whole training/testing process running
- [x] Think about data cleanup, tokenization and all the other shenanigans of working with SQuAD
    - [ ] Lowercasing
    - [ ] Dealing with abbreviations
    - [ ] Dealing with numbers, dates etc
- [x] Add encoding of character-level info as well as word-level info
- [ ] Make a distinction between train and non-train datasets for proper handling of char/word -> idx mappings (handled by wordvecs for now for word level but what about char level)
- [ ] Add unit testing for core components
- [ ] Make GPU compatible
- [ ] Debug simple architecture
- [ ] Implement BiDAF on top
- [ ] Add encoding of sentence-level info
- [ ] Integrate ELMo vectors
- [ ] Semi-supervised approach to make models resilient to single-word swaps
