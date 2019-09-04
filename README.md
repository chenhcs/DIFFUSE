# DIFFUSE

*DIFFUSE* is a deep learning based method for predicting isoform functions by integrating the data of isoform sequences, domains and expression profiles. This is an instruction of predicting isoform functions using *DIFFUSE*.

## Predicted Functions
- [Predicted functions](https://github.com/haochenucr/DIFFUSE/tree/master/results/all_predictions.txt) for all the 39,375 isoforms on 4,184 GO terms are saved in a text file. Redundancy in the GO predictions are removed. Considering the predicted functions of an isoform, all GO terms that have a child GO term assigned to the same isoform are discarded.

## Dependencies
- [Python 2.7.13](https://www.python.org/downloads/release/python-2713/)</br>
- [Keras](https://keras.io/)</br>
- [TensorFlow-GPU](https://www.tensorflow.org/)</br>
- [SciPy](https://www.scipy.org/)</br>

Set the backend of Keras as TensorFlow by modifying the [configuration file](https://keras.io/backend/).</br> 

## Data Preparation
- Download data from the [link](https://drive.google.com/file/d/1HkcRcGr9dNRaQfpohWHL3AX71OeG7Q0-/view?usp=sharing), unzip data.zip to the data/ folder.
- Preprocessing code for domain, sequence and expression data are provided in the [preprocessing](https://github.com/haochenucr/DIFFUSE/tree/master/preprocessing) directory, you can use them to process your own data.

## Get Started

### Test pre-trained models

- Pre-trained models for several GO terms are provided in the [saved_models](https://github.com/haochenucr/DIFFUSE/tree/master/saved_models) directory.
- Run the script `./codes/demo.sh` to generate predictions for the [test data](https://github.com/haochenucr/DIFFUSE/tree/master/data). You can change the GO term in this script to another one with a pre-trained model.
- Performance in terms of AUC and AUPRC will be reported. The predictions are saved in the [results](https://github.com/haochenucr/DIFFUSE/tree/master/results) directory. The first column in the file shows gene IDs, the second column shows isoform IDs and the third column shows prediction scores indicating how likely the corresponding isoforms have the GO term.

### Train new models

- Run the script `./src/train_new_model.sh` for training new models. You can change the GO term index in the script to train models for different GO terms appearing in the [GO term lists](https://github.com/haochenucr/DIFFUSE/tree/master/data/go_terms).

## Citation

If you find *DIFFUSE* is useful for your research, please consider citing the following paper:

	@article{chen2019diffuse,
	  title={DIFFUSE: predicting isoform functions from sequences and expression profiles via deep learning},
	  author={Chen, Hao and Shaw, Dipan and Zeng, Jianyang and Bu, Dongbo and Jiang, Tao},
  	  journal={Bioinformatics},
  	  volume={35},
  	  number={14},
  	  pages={i284--i294},
  	  year={2019},
  	  publisher={Oxford University Press}
	}
