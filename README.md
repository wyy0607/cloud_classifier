# Reproducible Model Development for Cloud Classification

Author: Yuyan Wu

# Table of Contents
* [Directory structure](#Directory-structure)
* [Data_Source](#Data-source)
* [Running the pipeline](#Running-the-pipeline)
* [Testing](#Testing)

## Directory structure 

```
├── README.md                         <- You are here
├── config                            <- Directory for configuration files 
│   ├── logging/                      <- Configuration of python loggers
│   ├── model_config.yaml             <- Configurations for model pipeline
│
├── data                              <- Folder that contains data used or generated. 
│   ├── interim/                      <- Intermidiate data generated during runing the model.
│   ├── raw/                          <- Raw data used for the model pipeline.
│
├── dockerfiles/                      <- Directory for all related Dockerfiles 
│   ├── Dockerfile                    <- Dockerfile for building image to execute run.py  
│   ├── Dockerfile.model              <- Dockerfile for building image to execute run.sh  
│   ├── Dockerfile.test               <- Dockerfile for building image to run unit tests
│
├── models/                           <- Trained model objects (TMOs), and model predictions, and/or model summaries
│
├── notebooks/
│   ├── clouds.ipynb                  <- Template notebook for analysis. 
│
├── src/                              <- Source data for the project.  
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── requirements.txt                  <- Python package dependencies 
├── run.py                            <- Simplifies the execution of one or more of the src scripts  
```

## Data source

The data used for this model pipeline comes from UCI Machine Learning Repository. To download the data,
you can go to this website 
https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data.
To acquire the raw data, The function `acquire_raw_data` under `src/create_datasets.py` 
can be used to get and save the raw data with specified input path (above link to the raw data) 
and output path (directory to save the raw data).  


## Running the pipeline 

#### Build the image 

Build Docker image for running tasks step by step, run from this directory (the root of the repo): 

```bash
docker build -f dockerfiles/Dockerfile -t clouds .
```

Build Docker image for running the entire model pipeline in a single command, 
run from this directory (the root of the repo): 

```bash
docker build -f dockerfiles/Dockerfile.model -t clouds_pipeline .
```

### Run each individual step of the model pipeline

#### Acquire the raw data and save it to the appropriate directory

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py get_raw_data
```

#### Create the cleaned dataset and save it to the appropriate directory

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py get_clouds
```

#### Generate the features and save them to the appropriate directory

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py generate_features
```

#### Generate the trained model object and save it to the appropriate directory

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py train_model
```

#### Produce predictions for evaluating the model and save them to the appropriate directory

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py score_model
```

#### Compute the performance metrics and save them to the appropriate directory

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds run.py evaluate
```

### Run entire pipeline

To run the entire model pipeline:

```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ clouds_pipeline run.sh
```

## Testing

Run the following to build Docker image for testing:

```bash
docker build -f dockerfiles/Dockerfile.test -t cloud-tests .
```

To run the tests, run: 

```bash
docker run cloud-tests
```
