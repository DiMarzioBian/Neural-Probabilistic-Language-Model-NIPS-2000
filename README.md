# A neural probabilistic language model

This is a pytorch implementation of NIPS'00 paper [A neural probabilistic language model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

************************************************************
********************   Training model   ********************
************************************************************
Run `python main.py`, this script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --path_data               location of the data corpus
  --num_worker              number of workers to load data
  --h_dim                   dimensions of hidden state and embedding vectors
  --optimizer               optimizer type: Adam, AdamW, RMSprop, Adagrad, SGD
  --lr                      initial learning rate
  --lr_step                 step length of lr decay
  --lr_gamma                strength of lr decay
  --epochs                  maximum epoch
  --batch_size              batch size
  --n_gram                  input sequence length
  --dropout                 dropout applied to layers (0 = no dropout)
  --skip_connect            enable skip connection between embedding and predictor
  --share_embedding         shared weights of embedding and predictor
  --share_embedding_strict  enabl predictor bias (not work if share_embedding=False)
  --seed                    random seed
  --es_patience_max         maximum patient of early stopping
  --save                    path to save the final model
  --onnx-export             ONNX_EXPORT path to export the final model in onnx format
  --device                  device to run code        
```
No arguments will run the model in the best settings achieved loss 5.0773.


************************************************************
*******************   Generating text   ********************
************************************************************
Run `python generate.py`, this script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --path_data               location of the data corpus
  --n_gram                  input sequence length
  --seed                    random seed
  --device                  device to run code 
  --out_f                   location of saving generated text
  --n_words                 number of generated words
  --checkpoint              lcoation of model checkpoint to use
  --temperature             temperature - higher will increase diversity
  --log-intervak            reporting interval    
```
Input text is hardcoded cannot be change by arguments because of length required is 40 minimum
if you are using the saved model.


************************************************************
*****************   Compute Correlation   ******************
************************************************************
Run `python correlation.py`, this script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --path_data               location of the data corpus
  --seed                    random seed
  --device                  device to run code 
  --path_data_new           location of the wordsim data corpus
  --checkpoint              lcoation of model checkpoint to use
```


************************************************************
********************   File structure   ********************
************************************************************

----P1_CE7455\
    |----correlation.py
    |----data\
    |    |----wikitext-2\
    |    |    |----README
    |    |    |----test.txt
    |    |    |----train.txt
    |    |    |----valid.txt
    |    |----wordsim353_sim_rel\
    |    |    |----wordsim_similarity_goldstandard.txt
    |----dataloader.py
    |----epoch.py
    |----generate.py
    |----main.py
    |----model.py
    |----README.md
    |----requirements.txt
    |----result\
    |    |----generated.txt
    |----saved_model\
    |    |----model.pt
