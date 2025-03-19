# Deploying a Self-Managed Kubernetes Cluster on Chameleon Instances

## Overview

This note details the steps to set up a Kubernetes cluster on Chameleon Cloud using OpenStack CLI, it was designed to handle microservices efficiently. It includes the following:

- **Kubernetes Cluster Setup:** Establishing the foundation for containerized applications.
- **Docker:** Packaging the application services to ensure portability.
- **Kubernetes Management:** Ensuring automatic scaling, load balancing, and high availability.
- **CI/CD Pipeline:** Automating the build, test, and deployment process.
- **Load Testing:** Simulating a load test to optimize performance.

The final system is a robust, scalable architecture that ensures reliable, minimal-intervention management of applications.

## Provision Resources Using the OpenStack CLI

Although the GUI is useful for exploring the capabilities of a cloud, the command line interface (CLI) is much more efficient for provisioning resources. This section covers the use of the OpenStack CLI to manage cloud resources.

### Objectives
- Provision server instances and ports using the OpenStack CLI.
- View already provisioned resources.

### Authentication
To use the CLI, authentication with the OpenStack Keystone service is required. Chameleon JupyterHub instance is already configured to authenticate the OpenStack client. Set the required environment variables by replacing `CHI-XXXXXX` with Chameleon project name:

```sh
export OS_AUTH_URL=https://kvm.tacc.chameleoncloud.org:5000/v3
export OS_PROJECT_NAME="CHI-XXXXXX"
export OS_REGION_NAME="KVM@TACC"
```

## Exploring the Cloud
To explore OpenStack capabilities, use:

```sh
openstack help
```

To list available services:

```sh
openstack catalog list
```

## Working with Network Resources

List all networks:

```sh
openstack network list
```

To filter the output by net ID:

```sh
openstack network list | grep netID
```

To get details of any network:

```sh
openstack network show private_cloud_net_netID
openstack network show sharednet1(the name of the network)
```

List subnets:

```sh
openstack subnet list | grep netID
```

Show subnet details:

```sh
openstack subnet show private_cloud_subnet_netID
```

### Creating Ports
Create two new ports:


```sh
openstack port create \
    --network private_cloud_net_netID \
    --fixed-ip subnet=private_cloud_subnet_netID,ip-address=192.168.1.12 \
    --disable-port-security \
    port2_netID

openstack port create \
    --network private_cloud_net_netID \
    --fixed-ip subnet=private_cloud_subnet_netID,ip-address=192.168.1.13 \
    --disable-port-security \
    port3_netID
```

List ports on the network:

```sh
openstack port list --network private_cloud_net_netID
```

## Working with Compute Resources

List provisioned servers that includes id in their names:

```sh
openstack server list --name "netID"
```

### Creating Compute Instances

List available images, flavors, and key pairs:

(A flavor in OpenStack defines the compute resources (vCPUs, RAM, and disk) allocated to a VM instance.)

```sh
openstack image list --limit 5
openstack flavor list
openstack keypair list
```

Launch new instances:

```sh
openstack server create \
  --image "CC-Ubuntu24.04" \
  --flavor m1.medium \
  --network sharednet1 \
  --port port2_netID \
  --security-group default \
  --security-group allow-ssh \
  --security-group allow-http-80 \
  --key-name id_rsa_chameleon \
  --user-data config-hosts.yaml \
  node2-cloud-netID

openstack server create \
  --image "CC-Ubuntu24.04" \
  --flavor m1.medium \
  --network sharednet1 \
  --port port3_netID \
  --security-group default \
  --security-group allow-ssh \
  --security-group allow-http-80 \
  --key-name id_rsa_chameleon \
  --user-data config-hosts.yaml \
  node3-cloud-netID
```

After launch, List new instances:

```sh
openstack server list --name "netID"
```

## Deploy a Service in a Docker Container

Docker is a platform for developing, shipping, and running applications in lightweight, portable containers that package code and dependencies together, ensuring consistency across environments.

### Objectives
- Pull and run Docker containers.
- Use `docker ps` to get details of running containers.
- Publish ports using `-p HOST_PORT:CONTAINER_PORT`.
- Understand Docker's overlay filesystem and storage mechanisms.

### Installing Docker

On `node1` host, install Docker engine:

```sh
sudo apt-get update
sudo apt-get -y install ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### Explanation of Commands:
- `apt-get update`: Refreshes the package list.
- `apt-get install -y`: Installs packages without prompting.
- `install -m 0755 -d /etc/apt/keyrings`: Creates a directory with appropriate permissions.
- `curl -fsSL ...`: Fetches and saves Docker’s GPG key.
- `chmod a+r ...`: Ensures the key file is readable.
- `echo ... | tee ...`: Adds Docker’s repository to package sources.

### Running a Test Container

Add the user to the Docker group:

```sh
sudo groupadd -f docker; sudo usermod -aG docker $USER
```

Exit and re-open the SSH session, then verify(should see a group of named docker):

```sh
id
```

Run a test container:

```sh
docker run hello-world
```

### Container Networking

Docker containers communicate using a virtual network. By default, Docker creates a bridge network (docker0), which acts like a virtual switch, allowing containers to talk to each other and access the internet while keeping them isolated from the host system.

Inspect Docker networking:

```sh
ip addr show docker0
docker network inspect bridge
```

Pull and run an Alpine container interactively:

```sh
docker pull alpine
docker run -it alpine
```
The `-it` flags in `docker run -it alpine` mean:
 
*   **`-i` (interactive)**: Keeps the container’s standard input (stdin) open, allowing you to interact with it.
    
*   **`-t` (TTY)**: Allocates a pseudo-terminal, making the shell inside the container behave like a normal terminal.

Inspect running containers (run on node1 host):

```sh
docker ps
```

Inspect network settings:

```sh
docker network inspect bridge
```

View host network settings:

```sh
ip addr
```

Inside the Alpine container, check network interfaces:

```sh
ip addr
```
![docker-bridge-structure](/img/docker-bridge.png)

Inside the container, get a list of network hops to address 1.1.1.1:

```sh
traceroute 1.1.1.1
```

to exit container session
```sh
exit
```

### Publishing a Port in Docker

First, run an Nginx container in detached mode:

```sh
# Run on node1 host
docker run -d nginx:alpine
```

- `-d`: Runs the container in "detached" mode (in the background).
- `nginx:alpine`: Uses the lightweight Alpine-based Nginx image.

The container starts an Nginx web server that listens on **port 80** inside the container.

#### Checking the Exposed Ports
To verify which ports the container exposes by default, run:

```sh
# Run on node1 host
docker image inspect nginx:alpine
```

Look for the `"ExposedPorts"` section in the output, which shows that **port 80** is exposed inside the container.

#### Accessing the Container Locally
Since the container is isolated, we need to find its internal IP address:

```sh
# Run on node1 host
docker network inspect bridge
```

Find the IP address of the running container (e.g., `172.17.0.X`). You can now test access using a terminal-based web browser:

```sh
# Run on node1 host
sudo apt -y install lynx  # Install lynx web browser
lynx http://172.17.0.X/  # Replace X with actual container IP
```
Use `q` and then `y` to quit the `lynx` browser.


This allows access **from the host**, but **not from an external machine**.

#### Mapping the Container's Port to the Host
To make the Nginx service accessible externally, stop the running container first:

```sh
# Run on node1 host
docker ps  # List running containers
docker stop CONTAINER_ID  # Stop the container
```

Now restart the container with port forwarding:

```sh
# Run on node1 host
docker run -d -p 80:80 nginx:alpine
```

- `-p 80:80`: Maps **port 80 of the host** to **port 80 of the container**, allowing external access.

#### Verifying Access
Check the host machine's IP address on the shared network:

```sh
# Run on node1 host
ip addr
```

Find the address of the form **10.56.X.Y**. Then, test access:

```sh
# Run on node1 host
lynx http://10.56.X.Y/
```

Now, since the instance has a **floating IP** assigned and the **security group allows traffic on port 80**, you can access the web server from an external browser using:

```sh
http://A.B.C.D/  # Replace A.B.C.D with your floating IP
```

You should see the Nginx welcome page.

#### Checking Firewall Rules
Docker sets up forwarding rules to enable this mapping. Verify them with:

```sh
# Run on node1 host
sudo iptables --list -n
sudo iptables --list -n -t nat
```

Look for the `DOCKER` chain, which includes additional rules to handle port forwarding.

#### Stopping the Container
To stop and remove the running container:

```sh
# Run on node1 host
docker ps  # List running containers
docker stop CONTAINER_ID  # Stop the container
```


### Container Filesystems


**Container Layer**: When a container starts, Docker adds a **read-write layer** on top of the **read-only image layers**. Changes (new, modified, or deleted files) occur only in this layer and are lost when the container stops unless committed to a new image. Multiple containers from the same image share the same image layers but have separate container layers, ensuring isolation while optimizing storage.

![docker container filesystem](/img/docker-filesys.png)

*Source: [Docker Documentation](https://docs.docker.com/engine/storage/drivers/overlayfs-driver/)*


To explore the Docker filesystem, First, we’ll start the container in detached mode:

```sh
# run on node1 host
docker run -d --name web1 nginx:alpine
```

- `-d`: Runs the container in detached mode (in the background).

Then, we’ll open a sh shell on the container in interactive (TTY) mode using docker exec:

```sh
# run on node1 host
docker exec -it web1 /bin/sh
```

- `-i`: Keeps the STDIN open even if not attached.
- `-t`: Allocates a pseudo-TTY.

If you now run:

```sh
# run inside web1 nginx container
df
```

inside the container, note that the root of the file tree (/) is on an overlay file system. The overlay file system is what makes containers so flexible and lightweight!

A Docker container image is made out of read-only image layers:

- The base layer
- Additional layers created by instructions used to build the container image, stacked on top of the base layer.

These layers are read-only and can be reused. When a container is created from an image, Docker adds a read-write container layer on top of those image layers. Changes are made in a staging area called the "workdir" before being copied to the container layer. However, these changes are temporary and last only as long as the container is running.

From the perspective of processes running inside the container, the filesystem appears as a merged version of the image layers and the container layer.

To examine this, run:

```sh
# run inside web1 nginx container
ls /
```

On the host machine, inspect the container’s filesystem:

```sh
# run on node1 host
docker inspect web1
```

In the "GraphDriver" section, you will see:

- "LowerDir": Paths representing image layers
- "UpperDir": The writable container layer
- "MergedDir": The merged view of all layers
- "WorkDir": A temporary working directory

We can save these paths in Bash variables:

```sh
# run on node1 host
LOWERDIRS=($(docker inspect web1 | jq -r '.[0].GraphDriver.Data.LowerDir' | tr ':' ' '))
UPPERDIR=$(docker inspect web1 | jq -r '.[0].GraphDriver.Data.UpperDir')
MERGED=$(docker inspect web1 | jq -r '.[0].GraphDriver.Data.MergedDir')
```

To examine the image layers:

```sh
# run on node1 host
for dir in "${LOWERDIRS[@]}"; do
    echo "$dir":
    sudo ls "$dir"
done
```

To inspect the writable container layer:

```sh
# run on node1 host
sudo ls $UPPERDIR
```

To inspect the merged view:

```sh
# run on node1 host
sudo ls $MERGED
```

### Editing Files in a Container

Inside the container, modify the index.html file:

```sh
# run inside web1 nginx container
vi usr/share/nginx/html/index.html
```

Change:

```html
<h1>Welcome to nginx!</h1>
```

to:

```html
<h1>Welcome to web1!</h1>
```
Switch to insert mode (`i`), delete "nginx," replace it with "web1," exit insert mode (`Esc`), then save and close the editor with `:wq` \+ `Enter`.


Save and exit `vi`. Test the change:

```sh
# run on node1 host
docker inspect web1
lynx http://172.17.0.X/
```

Check the changes in filesystem layers:

image layer:
```sh
# run on node1 host
for dir in "${LOWERDIRS[@]}"; do
     FILE="$dir/usr/share/nginx/html/index.html"
     sudo bash -c "[ -f '$FILE' ] && cat '$FILE'"
done
```

(writeable) container layer
```
sudo cat "$UPPERDIR/usr/share/nginx/html/index.html"
```
(processes inside the container will see)merged version
```
sudo cat "$MERGED/usr/share/nginx/html/index.html"
```

### Running Multiple Instances

Start a second instance:

```sh
# run on node1 host
docker run -d --name web2 nginx:alpine
```

- `-d`: Runs the container in detached mode.

Inspect its filesystem:

```sh
# run on node1 host
docker inspect web2
```

Stop both containers:

```sh
# run on node1 host
docker stop web1 web2
```



### Volume Mounts

To persist data beyond the lifetime of a container, create a volume:

```sh
# run on node1 host
docker volume create webvol
```

Run an nginx container with the volume mounted:

```sh
# run on node1 host
docker run -d -v webvol:/usr/share/nginx/html -p 80:80 nginx:alpine
```

- `-v`: Mounts a volume.
- `-p`: Maps container ports to host ports.

Modify the HTML file inside an alpine container:

```sh
# run on node1 host
docker run -it -v webvol:/data/web alpine
cd /data/web
vi index.html
```

- `-it`: Runs in interactive mode with a TTY.

Change:

```html
<h1>Welcome to nginx!</h1>
```

to:

```html
<h1>Welcome to docker volumes</h1>
```

Exit the container and check the changes persist.

### Bind Mounts

To directly modify files from the host, use a bind mount:

```sh
# run on node1 host
docker run -d -v ~/data/web:/usr/share/nginx/html -p 80:80 nginx:alpine
```

Edit the HTML file from the host:

```sh
# run on node1 host
sudo vim ~/data/web/index.html
```

press `i` to insert, Add:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Hello world</title>
  </head>
  <body>
    <p>Hello bind mounts</p>
  </body>
</html>
```

Restart the container and verify the changes persist.

### Building and Serving a Machine Learning Model

Clone the GourmetGram repository:

```sh
# run on node1 host
git clone https://github.com/teaching-on-testbeds/gourmetgram gourmetgram
```

Build the container image:

```sh
# run on node1 host
docker build -t gourmetgram-app:0.0.1 gourmetgram
```

Run the container:

```sh
# run on node1 host
docker run -d -p 80:8000 gourmetgram-app:0.0.1
```

- `-t`: Tags the image.
- `-p`: Maps ports.

Access the application in a browser:

```sh
http://A.B.C.D
```

Upload a food image to test the classification model.

## Deploy on Kubernetes
====================
 
### Introduction and Benefits
-------------------------
 
Kubernetes provides significant advantages over deploying single containers:
 
*   **Automatic failover**: When containers crash, Kubernetes replaces them automatically
    
*   **Load balancing**: Distributes traffic across multiple container instances
    
*   **Scalability**: Can dynamically adjust resources based on demand
    
*   **Orchestration**: Manages complex deployments across multiple hosts
    
 
### Setting Up the Environment
--------------------------
 
### SSH Configuration
 
First, we need SSH access to all nodes. From JupyterHub, open three terminals:
 
```sh
# bash

# Terminal 1: 
Connect to node1 (replace A.B.C.D with your floating IP)
ssh -A cc@A.B.C.D  # The -A flag enables agent forwarding for key authentication

# Terminal 2: Connect to node2 via node1 as a jump host
ssh -A -J cc@A.B.C.D cc@192.168.1.12  # -J specifies the jump host

# Terminal 3: Connect to node3 via node1
ssh -A -J cc@A.B.C.D cc@192.168.1.13
```
 
We use jump hosts because only node1 has a public IP address
 
### Preparing the Nodes
 
Before installing Kubernetes, we need to:
 
1.  Configure SSH between nodes for kubespray
    
2.  Disable firewalls
    
3.  Remove any existing Docker installations that might conflict
    
 
```sh
# bash
# On node1: Generate SSH key without password
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id\_rsa -q -N ""

# Copy keys to all nodes for passwordless access
ssh-copy-id -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id\_rsa.pub cc@192.168.1.11
ssh-copy-id -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id\_rsa.pub cc@192.168.1.12
ssh-copy-id -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id\_rsa.pub cc@192.168.1.13

# Disable firewalls on all nodes (run on each node)
sudo service firewalld stop

# Remove existing Docker on node1 (kubespray will install compatible version)
sudo apt -y remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo rm /etc/apt/sources.list.d/docker.list
sudo apt update
```
 
Installing Kubernetes with Kubespray
------------------------------------
 
Kubespray is a tool that leverages Ansible to deploy production-ready Kubernetes clusters.
 
### Setting Up Kubespray
 
```sh
# bash
# On node1: Clone kubespray repository
git clone --branch release-2.26 https://github.com/kubernetes-sigs/kubespray

# Install Python virtual environment and dependencies
sudo apt update
sudo apt -y install virtualenv
virtualenv -p python3 myenv
source myenv/bin/activate
cd kubespray
pip3 install -r requirements.txt
pip3 install ruamel.yaml
```
 
### Configuring the Cluster
 
The following steps create and customize the cluster configuration:
 
```sh

# bash
# Copy the sample inventory and modify configuration
cd
mv kubespray/inventory/sample kubespray/inventory/mycluster

# Configure Docker as the container manager (instead of containerd)
sed -i "s/container\_manager: containerd/container\_manager: docker/" kubespray/inventory/mycluster/group\_vars/k8s\_cluster/k8s-cluster.yml

# Enable the metrics server for horizontal pod autoscaling
sed -i "s/metrics\_server\_enabled: false/metrics\_server\_enabled: true/" kubespray/inventory/mycluster/group\_vars/k8s\_cluster/addons.yml

# Create inventory with our nodes
cd kubespray
declare -a IPS=(192.168.1.11 192.168.1.12 192.168.1.13)
CONFIG\_FILE=inventory/mycluster/hosts.yaml python3 contrib/inventory\_builder/inventory.py ${IPS\[@\]}

# Verify the configurationcat ~/kubespray/inventory/mycluster/hosts.yaml
```
 
### Deploying the Cluster
 
This step runs the Ansible playbook that configures all nodes and deploys Kubernetes:
 
```sh
# bash
# From node1:
cd
source myenv/bin/activate
cd kubespray
ansible-playbook -i inventory/mycluster/hosts.yaml --become --become-user=root cluster.yml
```
 
This process takes approximately 30 minutes. When complete, check the "PLAY RECAP" for any failure indicators (`failed=0` means success).
 
### Configuring kubectl Access
 
After installation, we need to set up kubectl for the non-root user:
 
```sh

# bash
cd
sudo cp -R /root/.kube /home/cc/.kube
sudo chown -R cc /home/cc/.kube
sudo chgrp -R cc /home/cc/.kube

# Verify cluster is working
kubectl get nodes  # Should show three nodes with "Ready" status
```
 
Setting Up a Container Registry
-------------------------------
 
Kubernetes needs access to container images across all nodes, so we'll create a private registry:
 
```sh

# bash
# On node1: Create Docker registry container
docker run -d -p 5000:5000 --restart always --name registry registry:2

# Configure all nodes to accept the insecure registry (run on each node)
sudo vim /etc/docker/daemon.json
# Add the following content:
{
	"insecure-registries": \["node1:5000"\]
}

# Restart Docker to apply changes (on each node)
sudo service docker restart

# Build and push our application image
docker build -t gourmetgram-app:0.0.1 gourmetgram
docker tag gourmetgram-app:0.0.1 node1:5000/gourmetgram-app:latest
docker push node1:5000/gourmetgram-app
```
 
Deploying the Application on Kubernetes
---------------------------------------
 
### Creating a Kubernetes Namespace
 
Namespaces provide logical isolation for applications:
 
```sh

# bash
kubectl create namespace kube-gourmetgram
kubectl get namespaces  # Verify the namespace was created
```
 
### Creating the Deployment Configuration
 
We'll define our application using Kubernetes manifests:
 
```sh

# bash
vim deployment.yaml
```
 
The configuration has three main parts:
 
1.  **Service**: Defines how the application is accessed externally
    
2.  **Deployment**: Specifies how pods are created and managed
    
3.  **HorizontalPodAutoscaler**: (Later) Controls automatic scaling
    
 
Build this configuration step by step:
 
```yaml

# 1. Service definition - network endpoint for our application
apiVersion: v1
kind: Service
metadata:
  name: gourmetgram-kube-svc
  namespace: kube-gourmetgram
spec:
  selector:
    app: gourmetgram-kube-app  # Connects to pods with this label
  ports:
    - protocol: "TCP"
      port: 80          # External port
      targetPort: 8000  # Internal container port
  externalIPs:
    - 10.56.X.Y  # Replace with your node1's public-facing IP
  type: ClusterIP

# 2. Deployment definition - manages the actual pods
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gourmetgram-kube-app
  namespace: kube-gourmetgram
spec:
  selector:
    matchLabels:
      app: gourmetgram-kube-app
  replicas: 1  # Start with a single instance
  template:
    metadata:
      labels:
        app: gourmetgram-kube-app
    spec:
      containers:
      - name: gourmetgram-kube-app
        image: node1:5000/gourmetgram-app:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        # Health check to determine pod readiness
        readinessProbe:
          httpGet:
            path: /test
            port: 8000
          periodSeconds: 5
          initialDelaySeconds: 5
          successThreshold: 3
        # Resource limits and requests
        resources:
          limits:
            cpu: "0.5"     # Maximum CPU allocation
            memory: "500Mi"  # Maximum memory allocation
          requests:
            cpu: "0.3"     # Minimum CPU requirement
            memory: "300Mi"  # Minimum memory requirement
```
 
### Deploying the Application
 
```sh
# bash
kubectl apply -f deployment.yaml

# Check deployment status
kubectl get all -n kube-gourmetgram -o wide
```
 
Watch for your pod to reach the "Running" state with "1/1" in the Ready column, indicating it passed the readiness probe.
 
### Testing the Application
 
Once deployed, access the application in your browser:
 
```
http://\[your-floating-IP\]
```
 
### Load Testing
 
Let's see how our single pod performs under load:
 
```sh
# bash
# Install load testing tool
sudo apt update
sudo apt -y install siege

# In terminal 1: Monitor resource usage
watch -n 5 kubectl top pod -n kube-gourmetgram

# In terminal 2: Generate load with 10 concurrent users for 30 seconds
siege -c 10 -t 30s http://$(curl -s ifconfig.me/ip)/test
```
 
 
Scaling Up with Multiple Replicas
---------------------------------
 
Improve performance by deploying multiple instances:
 
```sh
# bash

# Edit deployment to increase replicas
vim deployment.yaml
# Change replicas: 1 to replicas: 6

# Apply the change
kubectl apply -f deployment.yaml

# Verify multiple pods are running
kubectl get all -n kube-gourmetgram -o wide
```
 
### Testing with Multiple Replicas
 
Run the same load test as before:
 
```sh
# bash
# In terminal 1: Monitor all pods
watch -n 5 kubectl top pod -n kube-gourmetgram

# In terminal 2: Run the same load test
siege -c 10 -t 30s http://$(curl -s ifconfig.me/ip)/test
```
 
Compare the results with the single-pod test. 
 
*   Better throughput (more transactions)
    
*   Lower response times
    
*   Fewer connection failures
    
*   CPU load distributed across pods
    
 
Implementing Automatic Scaling
------------------------------
 
While having multiple replicas improves performance, it wastes resources during low-load periods. Let's implement horizontal pod autoscaling:
 
```sh
# bash
# Edit deployment.yaml
vim deployment.yaml

# 1. Change replicas back to 1
# 2. Add HorizontalPodAutoscaler configuration at the end:
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gourmetgram-kube-hpa
  namespace: kube-gourmetgram
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gourmetgram-kube-app
  minReplicas: 2  # Minimum number of pods
  maxReplicas: 6  # Maximum number of pods
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50  # Scale up when CPU usage exceeds 50%
```
 
Apply the configuration:
 
```sh
# bash
kubectl apply -f deployment.yaml

# Monitor the autoscaler and pods
kubectl get all -n kube-gourmetgram -o wide
```
 
### Testing Autoscaling
 
Test how the system automatically scales under load:
 
```sh
# bash

# In terminal 1: Monitor the deployment and HPA
watch -n 5 kubectl get all -n kube-gourmetgram -o wide

# In terminal 2: Generate load
siege -c 10 -t 30s http://$(curl -s ifconfig.me/ip)/test
```
 
Watch how Kubernetes:
 
1.  Detects increased CPU utilization
    
2.  Automatically increases the number of pods
    
3.  Distributes the load across the new pods
    
4.  Eventually scales back down after the load decreases (after ~5 minutes of low activity)
    
 
Cleanup
-------
 
```sh
kubectl delete -f deployment.yaml
```




