### Environment Setup
Just `pip install -r requirements.txt`. Only tested on python version 3.7.


### How the Code Works
Experiments are defined by json specification files in the `specs` directory. These control what hyperparameter grid to scan over in the data collection phase. Please don't alter them as they are currently set up to generate the same data as in the report.

The `runner` file contains the code for running the experiments. Simply type:
`python runner.py` and the experiment pipeline for both datasets will be run. This may take a couple hours, however; some models take quite a while to train, and each gets several runs through K-Fold Cross-Validation.
Results of the experiment will be stored in two new directories: `intention` and `pulsar`, representing the two datasets.

The `analyzer` file contains the code for generating plots and outputting the results. Simply run `python analyzer.py` _after_ running the `runner.py` file, and all the data collected during the data collection phase will be used to generate the relevant graphs. These graphs will be output to the `intention` and `pulsar` directories. The analyzer will also output the final model comparison table to `stdout`, which represent tables 11 and 12 in the report.

### Convenience Script
To run the whole project end-to-end, just run the command:
`python run.py`. This will run the `runner` and `analyzer` in sequence for both datasets. Results can be examined in the created folders `intention` and `pulsar`.
