An extensible, clean implementation of DocumentQA, and a basis for developing RCQA models

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Work plan:

- [x] Prepare harness for tokenization, batch building and evaluation
- [x] Make a basic LSTM->Dense->Spans&no-answer outputting model to get the whole training/testing process running
- [x] Think about data cleanup, tokenization and all the other shenanigans of working with SQuAD
    - [x] Lowercasing
    - [ ] Dealing with abbreviations
    - [ ] Dealing with numbers, dates etc
- [x] Add encoding of character-level info as well as word-level info
- [x] Add unit testing for core components
- [x] Make GPU compatible
- [x] Add option to read in a single answer span per question for training
- [x] Make a distinction between train and non-train datasets for proper handling of char/word -> idx mappings
- [x] Write dev validation during training
- [x] Implement BiDAF on top
- [x] Implement self attention as described in DocQA
- [x] Implement memory and runtime profiling
- [x] Add max context size
- [x] Do proper dropout
- [x] Test implementation with self attention
- [x] Do better structured config objects to pass around instead of bajillion parameters as it is used now
- [ ] Reproduce DocQA Performance
- [ ] Add UNK char vector for OOV chars
- [ ] Implement char CNN for char embeddings
- [ ] Add the option to output no-answer probabilities with the output
- [ ] Add encoding of sentence-level info
- [ ] Integrate ELMo vectors
