# medical-bert
This repo contains all the code for the software suite used to evaluate different NLP models.

It's mostly BERT.

## Configuration
All the classifiers are configurable from either a JSON file or can be over ridden via the command line.

Num_layers is a config value used to specify the number of encoding layers in the BERT Transformer. When this is set to 1 only one encodinng layer is included. When it is set to 12, which is the normal size for a BERT model, all encoding layers are included. Selecting a value greater than 12 will likely result in an out of bounds exception.
