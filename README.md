# Viterbi-Greedy-Decoding

Libraries used:
- Numpy
- Json
- operator

Within the current folder, have a subfolder called "data". Within the "data" subfolder, store the train, dev, and test files. This is because, in my program, I am reading the files using this path (paths are hardcoded). An example of the path is as follows: 'data/train'

The vocab.txt, hmm.json, greedy.out, and viterbi.out will be generated in the same folder where the Python file is located.


Simply run the entire Python program, the greedy and viterbi decoding accuracies on the dev file will be printed on the terminal and the required files i.e. vocab.txt, hmm.json, greedy.out, and viterbi.out will be generated in current folder.
