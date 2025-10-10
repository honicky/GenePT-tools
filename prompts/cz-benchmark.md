I want to create a notebook and/or code that approximates the benchmark at https://virtualcellmodels.cziscience.com/benchmark-task/cell-type-classification, and leverages code in https://github.com/chanzuckerberg/cz-benchmarks/blob/main/src/czbenchmarks/tasks/label_prediction.py where possible.

The benchmark measures the performance of embeddings on cell-type classification, and our MLP model is a classifier, so we will need to adapt the methodology slightly.                                                                                                                                               

The benchmark uses three different classifiers over several data sets and classifies cell type, do 5-fold cross validation on the data sets and take the average metric over the folds.  The approach I want to take is to take our model weights from our best model, freeze them, replace the output layer with an output layer that only outputs the cell types that actually exist in the dataset.  We could put another linear or non-linear layer in between, but that seems like it might over-parameterize, so I'm hesitant unless a new output layer by itself doen't work.  We would then train each fold for as many epochs as is helpful using a small holdout set from within the training fold.

In order to accomplish this, we will first need to embed the data using the GenePT embeddings.

Please create a spec that describes the implementation details, strategy for determining hyperparmaeters, training and measurement process. Reuse code from https://github.com/chanzuckerberg/cz-benchmarks/blob/main/src/czbenchmarks/tasks/label_prediction.py where possible.

Use @notebooks/cellxgene_v2_mlp.ipynb as a guide for the training process of the original model