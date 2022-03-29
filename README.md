This is the repository for quantum few-shot learning, a quantum embedding learning paradigm.
To prepare the environment, first install the `qtensor_ai` library at https://github.com/sss441803/QTensorAI.git. Then, install the code for `protonet` by running
```bash
python setup.py install
```
in the `QFSL` directory.
Before running the code, download the Omniglot dataset by running
```bash
bash download_omniglot.sh
```
To run the code, run
```bash
python scripts/train/run_train.py
```
See `scripts/train/run_train.py` for a list of parameters you can use to change the model definition and training parameters.