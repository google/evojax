# Running examples

EvoJAX comes with an extensive set of examples and tutorials in the form of Python scripts and notebooks.  
The examples and tutorials can be run readily on a machine with accelerators (GPUs/TPUs).  
In addition, we also provide instructions to run them on Google Cloud Platform (GCP) at the end of this README.

## Tutorials

We provide three tutorials for users who are interested in extending EvoJAX:
1. [Tutorial on Neuroevolution Algorithms](https://github.com/google/evojax/blob/main/examples/notebooks/TutorialAlgorithmImplementation.ipynb) introduces the `NEAlgorithm` interface and gives examples of wrapping an existing implementation or writing a new algorithm from scratch.
2. [Tutorial on Policies](https://github.com/google/evojax/blob/main/examples/notebooks/TutorialPolicyImplementation.ipynb) explains the `PolicyNetwork` and the `PolicyState` interfaces. It also describes how a user can use them to implement a new neural network policy.
3. [Tutorial on Tasks](https://github.com/google/evojax/blob/main/examples/notebooks/TutorialTaskImplementation.ipynb) describes the `VectorizedTask` and the `TaskState` interfaces. We show how various tasks such as MNIST classification and cartpole control can be easily implemented.

## Examples

Examples are in the form of Python scripts and notebooks.  
For scripts, detailed description and example commands to run the script can be found at the top of each file.  
For notebooks, users can simply open them in Google Colab (with GPU/TPU runtime) and run the commands in each cell.  

## Running EvoJAX on GCP

We recommend running the examples on a machine with modern accelerators (e.g., a VM on GCP) for the sake of performance.  
For this purpose, we give the instructions of running EvoJAX examples on a GCP VM.  

### Setting up a GCP Project
As a pre-requisite, you need a GCP project to run the example code. Please follow the setup [instructions](https://cloud.google.com/resource-manager/docs/creating-managing-projects?ref_topic=6158848&visit_id=637860227748301614-1502365940&rd=1) if you don’t have one already.

### Create a VM
Once you have the GCP project set up, use the following commands to create a virtual machine (VM) with one NVIDIA V100 GPU
(Feel free to change the number and type of GPUs).
When the VM is created, you will be able to ssh onto the machine. Upon first login, you may be asked if you would like to install Nvidia drivers.
Type “y” and wait until the installation completes.
```shell
# Configurations.
PROJECT=“your-project-name”  # Fill in your GCP project name.
ZONE=“desired-zone-name”     # E.g., us-central1-a
VM=“desired-vm-name”         # E.g., evojax-gcp

# Create a virtual machine.
gcloud compute instances create ${VM} \
--project=${PROJECT} \
--zone=${ZONE} \
--machine-type=n1-standard-8 \
--accelerator=count=1,type=nvidia-tesla-v100 \
--create-disk=auto-delete=yes,boot=yes,device-name=${VM},image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20220316-debian-10,mode=rw,size=50,type=projects/${PROJECT}/zones/${ZONE}/diskTypes/pd-balanced \
--maintenance-policy=TERMINATE

# Login to the VM.
gcloud compute ssh ${VM} --project=${PROJECT} --zone=${ZONE}
```

### Install Python Packages and Run Examples
When the drivers are installed, run the following commands to create a working directory and install JAX (with a GPU backend) for you.
```shell
# Create a working directory and install tools.
mkdir evojax_gcp
cd evojax_gcp
python3 -m venv venv
source venv/bin/activate
pip install -U pip

# Install JAX.
pip install --upgrade "jax[cuda]" -f \ https://storage.googleapis.com/jax-releases/jax_releases.html

# Test installation, confirm the output is a GPU device.
# Sample output: "[GpuDevice(id=0, process_index=0)]"
python -c "import jax; print(jax.devices())"
```

Finally, we are ready to install EvoJAX and run example codes.
```shell
# Download and install EvoJAX.
git clone https://github.com/google/evojax.git
cd evojax
pip install -e .

# CartPole: in about 2 min, you should see the average test score reaching 600+.
python examples/train_cartpole.py --gpu-id=0

# Multi-Dimensional Knapsack Problem: you should see approximated solution in less then 1 min.
python train_mdkp.py --gpu-id=0 --item-csv={items}.csv --cap-csv={caps}.csv  # With user CSV files.
python train_mdkp.py --gpu-id=0 --use-synthesized-data                       # Or with synthesized data.

# For notebook examples, install and start a Jupyter server.
pip install jupyter
jupyter notebook
```
