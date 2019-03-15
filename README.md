# DIFFUSE

This is a demo of predicting isoform functions using DIFFUSE.

## Dependencies
- [Python 2.7](https://www.python.org/downloads/release/python-2713/)</br>
- [Keras](https://keras.io/)</br>
- [TensorFlow-GPU](https://www.tensorflow.org/)</br>
- [SciPy](https://www.scipy.org/)</br>

Please set the backend of Keras as TensorFlow by modifying the [configuration file](https://keras.io/backend/), since the Pyramid Pooling layer (`./codes/layer/PyramidPooling.py`) is implemented following the TensorFlow dimension ordering convention.</br> 

## User Guide
- Install the dependencies and set the configuration as described abrove.
- Trained models for several GO terms are provided in the ./saved_models/ directory for demo use.
- Run the script `./codes/demo.sh` to generate predictions for the [test data](https://github.com/haochenucr/DIFFUSE/tree/master/data). The user can change the GO term in this script to another one with a trained model.
- Performance in terms of AUC and AUPRC will be reported. The predictions are saved in the ./results/ directory. The first column in the file shows gene IDs, the second column shows isoform IDs and the third column shows prediction scores indicating how likely the corresponding isoforms have the GO term.
- Note that the code and data released here are only for the demo use and the full version of DIFFUSE with its training data will be released soon.