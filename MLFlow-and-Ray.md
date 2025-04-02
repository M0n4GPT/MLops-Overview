# Training ML Models with MLFlow and Ray

## Overview

This experiment explores infrastructure and platform requirements for large-scale ML model training, focusing on following key components:

1. **Experiment tracking** using MLFlow
2. Use **Ray Train** for distributed training
3. Implement fault tolerance with checkpointing
4. Optimize resource usage with fractional GPUs


In this first part, we'll cover:
- Setting up a bare-metal GPU node on Chameleon Cloud
- Preparing a dataset for training
- Configuring and launching a complete MLFlow tracking server system
- Setting up a Jupyter notebook environment for ML experimentation


## Prerequisites 

## 1. Setting Up GPU Resources

We'll need a bare-metal node with GPUs. For the MLFlow section, a node with two GPUs is ideal to better understand distributed training logging. The Ray section specifically requires a node with two GPUs.

### Finding Suitable Hardware

First, we need to identify available GPU nodes on Chameleon:

1. Browse the [Chameleon Hardware Browser](https://chameleoncloud.org/hardware/)
2. Expand "Advanced Filters" and check the "2" box under "GPU count"
3. Click "View" to see suitable node types

For this use either:
- `gpu_mi100` nodes with two AMD MI100 GPUs
- `compute_liqid` nodes with two NVIDIA A100 40GB GPUs

### Creating a Lease


## 3. Preparing the Dataset

We use the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/) for ML training. First, prepare a Docker volume with this dataset.

```bash
# Create a volume to store the dataset
docker volume create food11

# Download and organize the dataset in the volume
docker compose -f mltrain-chi/docker/docker-compose-data.yaml up -d
```

The second command starts a temporary container that:
1. Downloads the Food-11 dataset
2. Organizes it in the Docker volume
3. Stops automatically when finished

To verify the container has completed its task:

```bash
# Check for running containers
docker ps
```

When there are no running containers, the data preparation is complete.

Verify the data structure:

```bash
# Inspect the volume contents using a temporary container
docker run --rm -it -v food11:/mnt alpine ls -l /mnt/Food-11/
```

Should see "evaluation", "validation", and "training" subfolders.

## 4. Setting Up the MLFlow Tracking Server

Now we configure a complete MLFlow tracking server system. This system consists of three main components:

1. **PostgreSQL database** - Stores structured data for each experiment run
2. **MinIO object store** - Stores artifacts like model weights and images
3. **MLFlow tracking server** - Provides the web interface and APIs

### Understanding the MLFlow Tracking Server Architecture

Our MLFlow tracking server uses Docker Compose to configure and launch multiple containers:

<!--![MLFlow experiment tracking server system](images/5-mlflow-system.svg)-->

The Docker Compose file includes:

- **Persistent volumes** for MinIO and PostgreSQL data
- **MinIO container** - Object storage with API on port 9000 and web UI on port 9001
- **MinIO bucket creation container** - Creates the storage bucket for MLFlow artifacts
- **PostgreSQL container** - Database backend for structured experiment data
- **MLFlow container** - The tracking server itself, accessible on port 8000

### Launching the MLFlow Tracking Server

To start the MLFlow system:

```bash
# Start all containers defined in the docker-compose file
docker compose -f mltrain-chi/docker/docker-compose-mlflow.yaml up -d
```

This command pulls the necessary images and starts all containers. To verify the system is running:

```bash
# Check running containers
docker ps
```

You should see `minio`, `postgres`, and `mlflow` containers running.

### Accessing the MLFlow Dashboards

Both MinIO and MLFlow provide web interfaces:

1. **MinIO Dashboard** (port 9001):
   - Open `http://<YOUR_FLOATING_IP>:9001` in a browser
   - Log in with:
     - Username: `your-access-key`
     - Password: `your-secret-key`
   - Explore the "Buckets" section to see the `mlflow-artifacts` storage bucket
   - Check "Monitoring > Metrics" to view storage system health

2. **MLFlow UI** (port 8000):
   - Open `http://<YOUR_FLOATING_IP>:8000` in a browser
   - You'll see the "Default" experiment (initially empty)

## 5. Setting Up the Jupyter Environment

Finally, we'll start a Jupyter server container for running experiments that will be tracked by MLFlow:

```bash
# Verify that the jupyter-mlflow image is available
docker image list
```

### For AMD GPU Nodes (gpu_mi100)

```bash
# Get the public IP address
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )

# Start the Jupyter container with AMD GPU support
docker run -d --rm -p 8888:8888 \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add $(getent group | grep render | cut -d':' -f 3) \
    --shm-size 16G \
    -v ~/mltrain-chi/workspace_mlflow:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-mlflow
```

### For NVIDIA GPU Nodes (compute_liqid)

```bash
# Get the public IP address
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )

# Start the Jupyter container with NVIDIA GPU support
docker run -d --rm -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/mltrain-chi/workspace_mlflow:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-mlflow
```

About these docker run commands:
- `-d`: Run the container in detached mode (background)
- `--rm`: Remove the container when it stops
- `-p 8888:8888`: Map container port 8888 to host port 8888
- GPU flag: Either `--device=/dev/kfd --device=/dev/dri` (AMD) or `--gpus all` (NVIDIA)
- `--shm-size 16G`: Increase shared memory size for ML training
- `-v ~/mltrain-chi/workspace_mlflow:/home/jovyan/work/`: Mount workspace directory
- `-v food11:/mnt/`: Mount the dataset volume
- `-e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/`: Set MLFlow tracking server URL
- `-e FOOD11_DATA_DIR=/mnt/Food-11`: Set dataset location

### Accessing Jupyter

To access the Jupyter notebook:

```bash
# Get the Jupyter access token
docker logs jupyter
```

Look for a URL like:
```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Open this URL in your browser, replacing `127.0.0.1` with your instance's floating IP address.

When Jupyter opens, navigate to the `work` directory in the file browser.

You can verify the environment setup by opening a terminal in Jupyter and running:

```bash
# View environment variables
env
```

Confirm that the `MLFLOW_TRACKING_URI` is set correctly with your floating IP address.

Now the environment is fully set up for ML experimentation with MLFlow tracking. 





# Tracking PyTorch Experiments with MLFlow

## Overview

Explore how to use MLFlow to track machine learning experiments, specifically focusing on PyTorch model training. MLFlow is an open-source platform that helps manage the entire machine learning lifecycle, including experimentation, reproducibility, and deployment.

Throughout this experiment, we will:

1. Run a baseline PyTorch training job without MLFlow tracking
2. Modify our PyTorch code to integrate MLFlow tracking capabilities
3. Run experiments with MLFlow tracking and analyze the results
4. Optimize our training process based on insights from MLFlow metrics
5. Register and version our models in the MLFlow Model Registry



## Setting Up the Environment

First, we need to clone the repository containing our training code:

```bash
# Navigate to the work directory
cd ~/work
# Clone the repository
git clone https://github.com/teaching-on-testbeds/gourmetgram-train
```

This repository contains a PyTorch training script for a food image classification model using the Food-11 dataset.

## Part 1: Running a Baseline PyTorch Training Job

Let's first run the original training script to understand how it works:

```bash
# Navigate to the repository directory
cd ~/work/gourmetgram-train
# Run the training script
python3 train.py
```

> **Note**: We won't let this training job finish as it would take a long time. After a few minutes, use `Ctrl+C` to stop the process. This is just to make sure everything works correctly.

This script trains a convolutional neural network that classifies food images into one of eleven categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. It uses a MobileNetV2 architecture as the base model with a custom classification head.

## Part 2: Integrating MLFlow Tracking

Now, switch to a branch that already has MLFlow tracking code integrated:

```bash
# Fetch all branches
git fetch -a
# Switch to the mlflow branch
git switch mlflow
```

To see the changes that were made to incorporate MLFlow tracking, we can compare the differences:

```bash
# View the differences between main and mlflow branches
git diff main..mlflow
```

> Press `q` to exit the diff view when you're done reviewing.

### Key MLFlow Additions

Here review the main changes that were made to integrate MLFlow tracking:

#### 1. Import MLFlow Libraries

```python
import mlflow
import mlflow.pytorch
```

MLFlow provides framework-specific modules for many ML frameworks, including PyTorch, scikit-learn, TensorFlow, and HuggingFace.

#### 2. Configure MLFlow

```python
mlflow.set_experiment("food11-classifier")
```

This sets the experiment name to "food11-classifier". In MLFlow, an experiment is a group of related runs, typically for a specific model or task.

> **Note**: We're using an environment variable `MLFLOW_TRACKING_URI` to specify the MLFlow server address instead of hardcoding it with `mlflow.set_tracking_uri()`.

#### 3. Start and End MLFlow Runs

```python
try: 
    mlflow.end_run()  # End pre-existing run, if there was one
except:
    pass
finally:
    mlflow.start_run(log_system_metrics=True)
```

Each training attempt is a "run" in MLFlow. The `log_system_metrics=True` argument enables automatic tracking of system metrics like CPU and GPU utilization.

#### 4. Log System Information

```python
mlflow.log_text(gpu_info, "gpu-info.txt")
```

This saves the GPU information as a text artifact in MLFlow.

#### 5. Log Hyperparameters

```python
mlflow.log_params(config)
```

This logs all the hyperparameters in our `config` dictionary, making it easy to track what settings were used for each run.

#### 6. Log Training Metrics

```python
mlflow.log_metrics(
    {"epoch_time": epoch_time,
     "train_loss": train_loss,
     "train_accuracy": train_acc,
     "val_loss": val_loss,
     "val_accuracy": val_acc,
     "trainable_params": trainable_params,
    }, step=epoch)
```

This logs various metrics for each training epoch, allowing us to track model performance over time.

#### 7. Log Model Checkpoints

```python
mlflow.pytorch.log_model(food11_model, "food11")
```

This saves the PyTorch model as an artifact in MLFlow, along with metadata about its dependencies and structure.

#### 8. Log Test Metrics

```python
mlflow.log_metrics(
    {"test_loss": test_loss,
     "test_accuracy": test_acc
    })
```

This logs the final evaluation metrics on the test set.

## Part 3: Running with MLFlow Tracking

Now run our training script with MLFlow tracking enabled:

```bash
# Run the training script with MLFlow tracking
python3 train.py
```

While the script is running, open the MLFlow UI in your browser by navigating to:
```
http://A.B.C.D:8000/
```
(Replace `A.B.C.D` with your instance's IP address)

In the MLFlow UI, you should see:
1. The "food11-classifier" experiment listed on the left
2. Your current run listed under this experiment
3. Parameters, metrics, and system information being logged in real-time

### Analyzing MLFlow Data

After the script has been running for a few minutes, check the "System metrics" tab in the MLFlow UI. Notice that the GPU utilization is low, indicating that our training process isn't efficiently using the GPU resources.


## Part 4: Optimizing Based on MLFlow Insights

Now improve training efficiency by increasing the number of data loader workers:

```bash
# Edit train.py to increase the number of data loader workers
# Change:
# train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
# to:
# train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=16)
# val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=16)
```

> **Explanation**: The `num_workers` parameter determines how many subprocesses to use for data loading. By increasing this value, we can load and preprocess data in parallel, which should keep the GPU busy and reduce waiting time.

Now let's commit our changes and run the training again:

```bash
# First, stop the current training run with Ctrl+C
# Configure git with your information
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"

# Commit the changes
git add train.py
git commit -m "Increase number of data loader workers, to improve low GPU utilization"

# View the recent commit history
git log -n 2

# Run the training script again
python3 train.py
```

In the MLFlow UI, find your new run and add a description to help you remember what changes you made:
> "Checking if increasing num_workers helps bring up GPU utilization."

After a few epochs, compare the two runs by looking at:
- GPU utilization (`gpu_0_utilization_percentage` under system metrics)
- Time per epoch (`epoch_time` under model metrics)

Should see a significant improvement in GPU utilization and reduced epoch time after your optimization.

## Part 5: Registering a Model in MLFlow

After your training completes, let's register the trained model in MLFlow's Model Registry:

1. In the MLFlow UI, navigate to your completed run
2. Click on the "Artifacts" tab
3. Find your model and click "Register model"
4. Select "Create new model" and name it `food11`

Now you can view your registered model in the "Models" tab, where you can see its lineage (the run that created it) and manage different versions as you create them.





# Ray Cluster for ML Training Jobs

## Overview


1. Set up a Ray cluster with head and worker nodes
2. Configure monitoring with Prometheus and Grafana
3. Submit jobs to the Ray cluster using different approaches
4. Explore Ray Train for distributed training
5. Test fault tolerance with checkpointing
6. Optimize hyperparameters with Ray Tune


## Setting Up the Ray Cluster

### Understanding the Ray Cluster Architecture

The overall system includes:
- A Ray head node for scheduling, management, and dashboard
- Two Ray worker nodes for computation
- Prometheus for metrics collection and Grafana for visualization
- MinIO object store for persistent storage
- A separate Jupyter notebook server for job submission

Ray provides components like Ray Cluster, Ray Train, Ray Tune, Ray Data, and Ray Serve. In this tutorial, we focus on Ray Cluster, Ray Train, and Ray Tune.

### Starting the Ray Cluster for AMD GPUs

First, verify you have two GPUs:

```bash
# run on node-mltrain
rocm-smi
```

Build a container image for Ray worker nodes:

```bash
# run on node-mltrain
docker build -t ray-rocm:2.42.1 -f mltrain-chi/docker/Dockerfile.ray-rocm .
```

Start the Ray cluster with Docker Compose:

```bash
# run on node-mltrain
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker compose -f mltrain-chi/docker/docker-compose-ray-rocm.yaml up -d
```

Verify that containers are running:

```bash
# run on node-mltrain
docker ps
```

Check that each worker sees a GPU:

```bash
# run on node-mltrain
docker exec ray-worker-0 "rocm-smi"
docker exec ray-worker-1 "rocm-smi"
```

### Starting the Ray Cluster for NVIDIA GPUs

Verify you have two GPUs:

```bash
# run on node-mltrain
nvidia-smi
```

Start the Ray cluster with Docker Compose:

```bash
# run on node-mltrain
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker compose -f mltrain-chi/docker/docker-compose-ray-cuda.yaml up -d
```

Verify containers are running:

```bash
# run on node-mltrain
docker ps
```

Check that each worker sees one GPU (the Docker Compose file assigns one GPU per worker):

```bash
# run on node-mltrain
docker exec -it ray-worker-0 nvidia-smi --list-gpus
docker exec -it ray-worker-1 nvidia-smi --list-gpus
```

### Starting a Jupyter Container

Build a Jupyter container to submit jobs to Ray:

```bash
# run on node-mltrain
docker build -t jupyter-ray -f mltrain-chi/docker/Dockerfile.jupyter-ray .
```

Run the Jupyter container:

```bash
# run on node-mltrain
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker run -d --rm -p 8888:8888 \
    -v ~/mltrain-chi/workspace_ray:/home/jovyan/work/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    --name jupyter \
    jupyter-ray
```

Get the Jupyter access token:

```bash
# run on node-mltrain
docker logs jupyter
```

Look for a URL like:
```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Access this URL in your browser, replacing `127.0.0.1` with your server's IP address.

Verify the `RAY_ADDRESS` environment variable is set correctly in a Jupyter terminal:

```bash
# runs on jupyter container inside node-mltrain
env
```

### Accessing the Ray Dashboard

Open the Ray dashboard in your browser:

```
http://A.B.C.D:8265
```

Replace `A.B.C.D` with your server's IP address. Check the "Cluster" tab to verify you see the head node and two worker nodes.

## Submitting Jobs to the Ray Cluster

### Submitting a Basic Job

First, clone the sample repository:

```bash
# run in a terminal inside jupyter container
cd ~/work
git clone https://github.com/teaching-on-testbeds/gourmetgram-train -b lightning
```

The runtime environment for our jobs needs to be specified. Two files are used:

1. `requirements.txt` - Lists required Python packages
2. `runtime.json` - Configures the environment:

```json
{
    "pip": "requirements.txt",
    "env_vars": {
        "FOOD11_DATA_DIR": "/mnt/Food-11"
    }
}
```

Submit the job to Ray:

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 8 --verbose --working-dir . -- python gourmetgram-train/train.py
```

This command:
- Specifies the runtime environment (`--runtime-env runtime.json`)
- Requests 1 GPU and 8 CPUs per job (`--entrypoint-num-gpus 1 --entrypoint-num-cpus 8`)
- Enables verbose output (`--verbose`)
- Packages the current directory for the worker (`--working-dir .`)
- Specifies the command to run (`python gourmetgram-train/train.py`)

Monitor the job in the Ray dashboard - it will transition from PENDING to RUNNING to SUCCEEDED.

### Testing Resource Constraints

Try requesting more resources than available:

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --entrypoint-num-gpus 2 --entrypoint-num-cpus 8 --verbose --working-dir . -- python gourmetgram-train/train.py
```

The job will remain in PENDING state because no worker has 2 GPUs. In a production environment with autoscaling enabled, Ray could spin up additional nodes to meet this demand.

Press Ctrl+C to stop waiting for the job.

### Using Ray Train

Switch to a version of code adapted for Ray Train:

```bash
# run in a terminal inside jupyter container
cd ~/work/gourmetgram-train
git stash # stash any changes you made to the current branch
git fetch -a
git switch ray
cd ~/work
```

The Ray Train version offers:
- Fault tolerance with checkpointing
- Distributed training across workers
- Integration with Ray Tune for hyperparameter optimization

Submit the Ray Train job:

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --working-dir . -- python gourmetgram-train/train.py
```

Note that we don't specify GPU/CPU requirements here because they're defined in the script's `ScalingConfig`.

### Scaling to Multiple Workers

Edit `train.py` to use multiple workers by changing:

```python
scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
```

to:

```python
scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
```

Submit the job:

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --working-dir . -- python gourmetgram-train/train.py
```

Monitor GPU usage with:

```bash
# runs on node-mltrain
nvtop
```

You should see both GPUs being utilized.

### Using Fractional GPUs

Ray allows requesting fractional GPU resources for better resource utilization. Change:

```python
scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
```

to:

```python
scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 0.5, "CPU": 4})
```

Open three terminals and submit the job in each:

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
cd ~/work
ray job submit --runtime-env runtime.json --working-dir . -- python gourmetgram-train/train.py
```

Monitor GPU usage:

```bash
# runs on node-mltrain
nvtop
```

You should see one GPU handling two jobs and another handling one job. This approach increases cluster throughput for jobs that don't fully utilize a GPU.

### Testing Fault Tolerance

Switch to a version with fault tolerance:

```bash
# run in a terminal inside jupyter container
cd ~/work/gourmetgram-train
git stash # stash any changes you made to the current branch
git fetch -a
git switch fault_tolerance
cd ~/work
```

Submit the job:

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --working-dir . -- python gourmetgram-train/train.py
```

Wait until about 10 epochs have completed, then identify which GPU is running the job:

```bash
# runs on node-mltrain
nvtop
```

Simulate a worker failure by stopping the container:

```bash
# runs on node-mltrain
# Use one of these commands depending on which worker is running your job:
# docker stop ray-worker-0
# docker stop ray-worker-1
```

Observe in `nvtop` that the job transfers to the other GPU, and in the job logs that it resumes from a checkpoint.

After the job finishes, restart the worker:

```bash
# runs on node-mltrain
# docker start ray-worker-0
# docker start ray-worker-1
```

### Using Ray Tune for Hyperparameter Optimization

Switch to the Ray Tune version:

```bash
# run in a terminal inside jupyter container
cd ~/work/gourmetgram-train
git stash # stash any changes you made to the current branch
git fetch -a
git switch tune
cd ~/work
```

Submit the job:

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --working-dir . -- python gourmetgram-train/train.py
```

This version uses the ASHA scheduler, which automatically terminates less promising configurations, saving cluster resources compared to grid search or random search.

## Stopping the Ray Cluster

When finished, stop the Ray cluster:

For AMD GPUs:
```bash
# run on node-mltrain
docker compose -f mltrain-chi/docker/docker-compose-ray-rocm.yaml down
```

For NVIDIA GPUs:
```bash
# run on node-mltrain
docker compose -f mltrain-chi/docker/docker-compose-ray-cuda.yaml down
```

Stop the Jupyter server:
```bash
# run on node-mltrain
docker stop jupyter
```

## Result Summary
Based on the "Jobs" section of the Ray cluster dashboard at the end of the experiment
* The baseline runtime for first Ray Train job is 6m23s. 
* The job that ran on two worker nodes sped up training a lot because it completed in 4m3s.
* The runtimes of the three jobs were 8m2s, 8m4s, and 6m18s, and the start time of earlier job to end time of last job is 8m29s. If these jobs had run sequentially using the baseline runtime it will take 19m9s.
* Relative to the baseline runtime, the interrupted job took 7m9s. It is 46s longer than the baseline.If instead of automatic fault tolerance, it would have taken about 6m23s + 3m12s = 9m35s when I had completed 50% of the job.



