
from textgenrnn import textgenrnn

textgenrn = textgenrnn(weights_path='showerthoughts_weights.hdf5', vocab_path='showerthoughts_vocab.json', config_path='showerthoughts_config.json')

textgenrn.generate(1, temperature=0.5)
textgenrn.generate(1, temperature=0.5)
