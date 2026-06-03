pytorch-training-assessment 



Environment Setup

Option 1: Kaggle via VSCode 



You can set up and connect Kaggle environments in Visual Studio Code (VSCode) using two primary methods: connecting directly to a live Kaggle Notebook's remote Jupyter server or replicating the exact Kaggle Environment locally using Docker. 



Method 1: Connect to a Live Kaggle GPU/CPU Kernel (Remote Execution)



This method allows you to use Kaggle's cloud compute (including free GPUs) while coding in side your local VS Code interface. 



Step 1: Initialise your Kaggle Notebook 

1. Go to your Kaggle account, open or create a new notebook, and start the session
2. (Optional) Turn on GPU or TPU acceleration in the right hand settings panel if your workload requires it. 



Step 2: Extract the Remote URL 

1. In the Kaggle notebook top menu bar click on Run -> Run Jupyter Server. 
2. A side panel will appear. Copy the unique VS Code Compatible URL provided there. 



Step 3: Configure VS Code

1. Open Visual Studio Code and make sure you have the official Jupyter Extension installed
2. Create or open an .ipynb notebook file inside VS Code
3. Click on Select Kernel in the top-right corner of the notebook editor. 
4. Select Existing Jupyter Server from the dropdoen editor
5. Paste the Kaggle URL you copied in step 2 into the text prompt field and press enter
6. Choose the newly loaded remote kernel to run your local cells directly on Kaggle infrastructure



*Note: If the Kaggle kernel shuts down due to inactivity, you will need to start a new session and update the Jupyter Server URL in VS Code.* 







Method 2: Duplicate the Kaggle Environment Locally (via Docker) 



If you prefer offline coding or want a persistant local copy of Kaggle's vast package stack, you can run Kaggle's official Docker containers 



Step 1: Install Prerequisites

1. Download and install Docker Desktop on your machine.
2. Install the Dev Containers Extension in VS Code 



Step 2: Download the Kaggle Docker Image

Open your local system terminal or command prompt and pull down Kaggle’s official pre-built Python environment image: 



For CPU workflows: 



docker pull gcr.io/kaggle-images/python



For GPU workflows (Requires NVIDIA Container Toolkit): 



docker pull gcr.io/kaggle-gpu-images/python 



Step 3: Run and Attach via VS Code

1. Launch the container locally by executing the following terminal command (replace the image tag if using the GPU version):



docker run --name kaggle-local -v /path/to/your/project:/home/kaggle -it gcr.io/kaggle-images/python /bin/bash



2\. Open VS Code, press Ctrl+Shift+P (or Cmd+Shift+P on Mac) to bring up the Command Palette.

3.Select Dev Containers: Attach to Running Container... 

4.Choose kaggle-local from the list



VS Code will configure a dedicated workspace instance operating inside the official, completely mirrored Kaggle software runtime





Tip: Integrating Kaggle Datasets directly into VS Code

Regardless of the environment choice above, you can stream competition files and datasets right into your project directory using the official Kaggle CLI tool: 



1. Go to your Kaggle Profile -> Settings -> Click Create New API Token to download your personal Kaggle.json credential file
2. Move that file to your local computers home directory under a hidded folder named .Kaggle (e.g. \~/.Kaggle/Kaggle.json on macOS/Linux or C:\\Users\\<User>\\.kaggle\\kaggle.json on Windows).
3. Within your VS code terminal, install the library:

&#x20;     pip install Kaggle 

4\. Find any dataset online and click Copy API Command from the dataset options menu. run that command in your VS Code terminal to programmatically download files instantly into your workplace. 









Option 2: Azure VM via SSH

Setting up environments on an Azure Virtual Machine (VM) via SSH involves opening Port 22 on Azure, downloading your private .pem key and using a local terminal to connect and run your environment installation scripts



1. Verify Azure Network Settings

Before trying to connect, you must make sure that Azure allows incoming SSH traffic. 



1. Open the Azure Portal
2. Navigate yo your Virtual Machine and select Networking under the Settings menu on the left
3. Check the Inbound port rules. Ensure there is a rule allowing SSH traffic on Port 22
4. If it is missing, click Add inbound port rule, set Service to SSH and click Add
5. Go to the VM Overview page and copy the Public IP address 



2\. Set Local File Permissions (Mac/Linux Only)

If you generated a new key pair during the Azure VM creation process, you downloaded a private key file (e.g. myKey.pem). If you are using a Mac or Linux terminal, you must restrict its permissions: 



chmod 400 /path/to/your/myKey.pem



(Windows users using standard Command Prompt or PowerShell can skip this step) 



3\. Connect via SSH

Open your local computer's terminal (Command Prompt, PowerShell or macOS Terminal) and type the following command to establish a connection: 



ssh -i /path/to/your/myKey.pem azureuser@<Your-VM-Public-IP>





* Replace  /path/to/your/myKey.pem with your actual file path
* Replace azureuser with the administrator username you designated during VM setup
* Replace <Your-VM-Public-IP? with the IP address copied from Azure



If prompted with a warning about host authenticity, type yes and press Enter





4\. Set Up the Environment

Once your terminal prompt changes to something like azureuser@your-vm-name:\~$, you are officially inside the Azure VM. You can now build out your chosen environment.



Option A: Python Virtual Environment

For data science, automation, or Python web apps, initialize your environment by executing the following commands sequentially: 



\# Update the system package repository

sudo apt update \&\& sudo apt upgrade -y



\# Install Python and the virtual environment package

sudo apt install python3-pip python3-venv -y



\# Create a new environment directory named 'myenv'

python3 -m venv myenv



\# Activate your new environment

source myenv/bin/activate



Option B: Docker Container Environment

For microservices or deploying pre-configured tech stacks, install Docker directly to manage isolated containers: 



\# Install Docker

sudo apt update

sudo apt install docker.io -y



\# Start and enable the Docker service

sudo systemctl start docker

sudo systemctl enable docker



\# Allow your user account to run Docker commands without sudo

sudo usermod -aG docker $USER



(Note: You will need to type exit to disconnect, then SSH back in for the user group changes to take effect) 



Option C: Node.js Environment

For hosting web applications or JavaScript APIs:



\# Download and install NodeSource Node.js setup

curl -fsSL https://nodesource.com | sudo -E bash -



\# Install Node.js and NPM

sudo apt install -y nodejs



5\. Finalise App Network Rules (If Applicable)

If your environment hosts a service (like a web server on Port 80, a Node app on Port 3000, or a container on Port 8080), you must return to the Azure Portal -> Networking page. Create another Inbound port rule matching the specific port number your application uses so it can receive internet traffic. 





Sample Commands With Expected Outputs  



Kaggle Via VS Code 



1\. Starting the Tunnel (Kaggle Cell) 

When you run the setup script in your Kaggle notebook, the logs will prompt you to authenticate.



Command: 

!./code tunnel --name kaggle-env



Expected Output: 

\*

\* Visual Studio Code Server

\*

\* By using the software, you agree to the License Terms and Privacy Statement.

\*

To grant access to the server, please open the following URL and enter the code below:

&#x20; URL: https://github.com

&#x20; Code: A1B2-C3D4



Note: Once you enter that 8 digit code on GitHub, the Kaggle cell output will automatically refresh to show: 

\[2026-06-03 13:42:01] info Open this link in your browser to connect to this machine: https://vscode.dev

\[2026-06-03 13:42:02] info Tunnel is ready!



2\. Verifying the Remote Environment (VS Code Terminal) 

&#x20;Once connected via your local VS Code application, open a new terminal (Ctrl + \~). Run these commands to verify you are truly using Kaggle's hardware and file system.



Check GPU Availability



Verify that VS Code can see Kaggle's Nvidia graphics cards.



Command:

nvidia-smi



Expected Output: 

+-----------------------------------------------------------------------------+



| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |

|-------------------------------+----------------------+----------------------+



| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |

| Fan  Temp  Perf  Pwr:Usage/Cap|         Maut | Compute M. |

|===============================+======================+======================|

|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |

+-------------------------------+----------------------+----------------------+ 



Check Active Python Paths

Verify that your terminal is using Kaggle's pre-configured Conda environment.



Command:

which python



Expected Output: 

/opt/conda/bin/python 



View Available Datasets

Check if your attached Kaggle datasets are visible in your file tree.



Command: 

ls -la /kaggle/input/



Expected Output:

total 12

drwxr-xr-x 3 root root 4096 Jun  3 13:40 .

drwxr-xr-x 1 root root 4096 Jun  3 13:40 ..

drwxr-xr-x 2 root root 4096 Jun  3 13:40 titanic-dataset  



3\. Testing Code Execution (VS Code Notebook Cell)

Create a test cell in your VS Code notebook interface to ensure libraries can utilize the GPU hardware.



Command:

import torch

print("PyTorch Version:", torch.\_\_version\_\_)

print("GPU Available:", torch.cuda.is\_available())

if torch.cuda.is\_available():

&#x20;   print("Device Name:", torch.cuda.get\_device\_name(0)) 



Expected Output: 

PyTorch Version: 2.1.2

GPU Available: True

Device Name: Tesla T4





Azure VM via SSH

1\. Connecting via SSH

This command initiates the connection from your local computer to the remote Azure VM: 

ssh -i \~/Downloads/azure\_key.pem azureuser@40.112.50.12 



Expected Output: 

The authenticity of host '40.112.50.12 (40.112.50.12)' can't be established.

ED25519 key fingerprint is SHA256:7uX9jK...

This key is not known by any other names.

Are you sure you want to continue connecting (yes/no/\[fingerprint])? yes

Warning: Permanently added '40.112.50.12' (ED25519) to the list of known hosts.

Welcome to Ubuntu 24.04 LTS (GNU/Linux 6.8.0-1008-azure x86\_64)



&#x20;\* Documentation:  https://ubuntu.com

&#x20;\* Management:     https://canonical.com

&#x20;\* Support:        https://ubuntu.com



Expanded Security Maintenance for Applications is not enabled.



azureuser@azure-web-vm:\~$ 



2\. Updating System Packages

This command syncs your VM's package manager with the latest software repositories:

sudo apt update



Expected Output:

Hit:1 http://ubuntu.com noble InRelease

Get:2 http://ubuntu.com noble-updates InRelease \[126 kB]

Get:3 http://ubuntu.com noble-backports InRelease \[126 kB]

Get:4 http://ubuntu.com noble-security InRelease \[126 kB]

Fetched 378 kB in 1s (420 kB/s)

Reading package lists... Done

Building dependency tree... Done

Reading state information... Done 



3\. Setting Up Python venv 

These commands install Python utilities, create an isolated directory, and turn it on: 

\# 1. Install pip and virtual environment packages

sudo apt install python3-pip python3-venv -y



\# 2. Create the environment

python3 -m venv myenv



\# 3. Activate the environment

source myenv/bin/activate 



Expected Output: 

(myenv) azureuser@azure-web-vm:\~$ 





4\. Setting Up Docker

These commands install Docker and verify that the background engine is running smoothly: 

\# 1. Install Docker engine

sudo apt install docker.io -y



\# 2. Check the system process status

sudo systemctl status docker



Expected Output: 

● docker.service - Docker Application Container Engine

&#x20;    Loaded: loaded (/usr/lib/systemd/system/docker.service; enabled; preset: enabled)

&#x20;    Active: active (running) since Wed 2026-06-03 13:58:12 UTC; 12s ago

TriggeredBy: ● docker.socket

&#x20;      Docs: https://docker.com

&#x20;  Main PID: 4321 (dockerd)

&#x20;     Tasks: 8

&#x20;    Memory: 28.4M



5\. Verifying Your Software Versions

Once any environment installation is complete, running version checks confirms the software is properly linked to your environment path variables: 

node -v

python3 --version

docker --version



Expected Output: 

v20.11.0

Python 3.12.3

Docker version 24.0.7, build 24.0.7-0ubuntu4



Troubleshooting for Common Issues 

Azure VM via SSH 



1\. Error: "Connection timed out" or "Port 22: Connection refused

"You try to run the SSH command, but the terminal hangs for 30 seconds and fails.



The Cause

The Azure Network Security Group (NSG) is blocking your connection, or you are using the wrong IP address.



How to Fix

Log into the Azure Portal.

Go to your VM page and verify the Public IP address has not changed (Dynamic IPs change if the VM is stopped and restarted).

Click on Networking (under Settings).Verify there is an Inbound Port Rule named AllowSSH. 

Ensure its settings are: Source: Any (or your specific local public IP), Destination Port: 22, Protocol: TCP, Action: Allow. 





2\. Error: "Permissions 0644 for 'key.pem' are too open"

The SSH connection is immediately rejected by your local terminal with a warning about unprotected private key files.



The Cause 

Mac and Linux operating systems reject SSH keys that can be read by other user accounts on your local computer. 



How to Fix

Run this command in your local terminal to restrict file access strictly to your user account:

chmod 400 \~/Downloads/azure\_key.pem





Expected Outcome: The command runs silently with no output. You can now retry your SSH connection command successfully. 



3\. Error: "Permission denied (publickey)" 

The VM active rejection message appears immediately after you run the SSH command. 



The Cause

You are using the wrong username, or the wrong .pem private key file for this specific VM. 



How to Fix

Check the username: Azure defaults to the username you typed when creating the VM (e.g., azureuser, ubuntu, or your custom name). It is never root.



Verify the key matches: If you have multiple Azure keys, confirm you are pointing to the correct file path. You can verify your current local username configuration inside Azure under the Reset password tab on the VM menu. 



4\. Error: "Package 'python3-venv' has no installation candidate" 

When running sudo apt install python3-venv, Ubuntu claims the package does not exist.



The Cause

The VM's local package repository list is outdated and does not know where to download the package from.



How to Fix

You must force a repository sync before installing new tools:



sudo apt update

&#x20;

Expected Output: A list of Get: and Hit: URLs appearing on screen. Once completed, re-run your sudo apt install python3-venv -y command. 



5\. Error: "permission denied while trying to connect to the Docker daemon socket"

You installed Docker, but running docker ps or docker run outputs a permission error.



The Cause

By default, Docker commands require root privileges (sudo).



How to Fix

Add your Azure user to the Docker Linux group so you do not have to type sudo every time:

sudo usermod -aG docker $USER



Crucial Step: You must close your connection by typing exit, then log back into the VM via SSH. The security group changes will not apply until you start a new terminal session. 



6\. Issue: Application is running on the VM, but cannot be accessed via the web browser

Your Python, Node, or Docker app is running perfectly on port 8080 inside the terminal, but typing <VM-IP-Address>:8080 into your web browser times out. 



The Cause

The application is running inside the VM, but Azure's firewall is blocking outside internet traffic from reaching that specific port. 



How to Fix

Go to Azure Portal -> your VM -> Networking.

Click Add inbound port rule.

Set Destination port ranges to the exact port your app uses (e.g., 8080, 3000, or 80).

Set Protocol to TCP and Action to Allow. 

Click Add and refresh your web browser after 60 seconds.



Kaggle Via VS Code: 

1\. Connection Timeout or "Tunnel Disconnected"

Symptoms: VS Code suddenly loses connection, or the Kaggle cell finishes running on its own.



The Cause: Kaggle notebooks automatically shut down if there is no user activity inside the Kaggle browser tab for 15–20 minutes, even if you are actively coding inside VS Code.



The Fix: Keep your Kaggle browser tab open in the background. To completely prevent timeouts, paste and run this JavaScript snippet inside your browser console (F12) while on the Kaggle page to simulate activity:



setInterval(() => { 

&#x20;   console.log("Keeping Kaggle alive"); 

&#x20;   window.dispatchEvent(new Event('refresh')); 

}, 60000);



2\. "Command not found" or permission errors

Symptoms: Running ./code tunnel returns bash: ./code: No such file or directory or Permission denied.



The Cause: The VS Code CLI binary did not extract correctly, or it lost its executable file permissions.



The Fix: Ensure you are in the correct directory and manually grant permission by running:

chmod +x ./code

./code tunnel --name kaggle-env



3\. Port 80 / 443 Blocked or Proxy Failures

Symptoms: The CLI hangs indefinitely at Starting tunnel... or throws a network connection error.



The Cause: Kaggle's internal network restrictions are blocking the connection, or you forgot to toggle the internet switch.



The Fix: Look at the right-hand settings panel in Kaggle. Ensure Internet on is toggled active. You cannot download the CLI or host a tunnel without this.



If it still fails, use the --accept-server-license-terms flag to bypass interactive prompts:

./code tunnel --accept-server-license-terms --name kaggle-env





4\. VS Code Notebook "Missing Kernel" or "Ipynb Extensions Required"

Symptoms: You open a notebook inside VS Code, but you cannot run cells, or /opt/conda/bin/python is missing.



The Cause: Your local VS Code needs extensions installed inside the remote tunnel environment, not just on your local machine.



The Fix:Click on the Extensions tab (Ctrl+Shift+X).Look for the section titled SSH / Tunnel: Kaggle (or your tunnel name).

Click Install in Tunnel for both the Python and Jupyter extensions. 



5\. Out of Memory (OOM) Crashes

Symptoms: The VS Code terminal crashes instantly when you start training a model or loading a large dataset.



The Cause: VS Code and its background extensions consume roughly 1–2 GB of RAM, leaving less memory for your Kaggle notebook (which caps at 16GB or 30GB depending on the instance).



The Fix: Avoid using memory-heavy VS Code extensions like large language model autocompletes (e.g., Copilot) inside the remote environment. Clean up your RAM inside your script using:



import gc

import torch

gc.collect()

torch.cuda.empty\_cache()



Environment requirements and Dependencies 



Kaggle vis VS Code



When using the VS Code Remote Tunnels Method, you do not need to install complex machine learning frameworks from scratch. Your environment automatically inherits Kaggle's massive, pre-configured production environment. 



1. The Base Environment (What is actually there) 



Kaggle's default environment runs on a specialised Linux-based Docker image. By targeting the /opt/conda/bin/python interpreter in VS Code, you instantly get access to hundreds of pre-installed libraries: 



* Core ML Frameworks: PyTorch, TensorFlow, JAX
* Data Processing: Pandas, NumPy, Scikit-learn, Polars
* GPU Acceleration: CUDA Toolkit (pre-matched to the assigned GPU), cuDNN 



2\. Required VS Code Extensions (Installed Locally) 



To Establish the connection and run code, your local VS Code application must have these three extensions installed from the marketplace. 



* Remote- Tunnels (ms.vscode.remote.server)- Mandataes to bridge your PC to Kaggle
* Python (ms-python.python)- Adds syntax support and environment detection
* Jupyter(ms.toolsai.jupyter)- Enables running .ipynb notebook calls directly in Vs Code



3\. Installing Custom Dependencies



If your project requires a specific library not included by Kaggle (like a niche Hugging Face PAckage or a specific utility), you can install it on the fly



Open a terminal inside your connected VS Code window and use Kaggle's isolated Conda package manager: 



\# Install a standard package via pip

/opt/conda/bin/pip install accelerate



\# Install a specific version of a library

/opt/conda/bin/pip install transformers==4.40.0



Note: Avoid using raw pip install without the full path, as it might default to a user-directory outside your active Conda environment



4\. Saving Dependencies (requirements.txt) 



Because Kaggle instances are completely ephemeral (wiped entirely clear whenever the notebook session stops), any extra libraries you install will vanish when you disconnect. 



To make your workplace reproducible: 



1. Create a requirements.txt file in your VS Code workspace folder (/Kaggle/working). 
2. Add your custom packages to it: 



transformers==4.40.0

accelerate>=0.28.0

wandb



3\. Update your initial Kaggle setup cell so it automatically reinstalls your custom toolkits every time you boot the notebook up: 



\# Run this right after starting the tunnel to restore your workspace libraries

!/opt/conda/bin/pip install -r /kaggle/working/requirements.txt



Azure VM via SSH 



Before installing packages, your Azure VM needs specific system recourses, basic development tools and language specific dependency managers to build and run code successfully.  



1. Hardware \&  System Requirements



Ensure your chosen Azure VM size matches the workload requirements for your environment: 



* Python/node.js Development: Minimum Standard\_B1s (1 vCPU, 1 GB  RAM). 
* Docker Environments: Minimum Standard\_B2s (2 vCPU, 4 GB RAM). Docker engines and multiple running containers will crash on 1GB RAM due to Out-Of-Memory (OOM) errors. 
* Storage Check: Run df -h to verify you have disk space. Ubuntu installations usually require at least 10-20 GB of OS disk space 



2\. Core System Dependencies



Many Pthon packages (lie cryptography or numpy) and Node.js native modules compile code during installation. They will fall if your VM lacks standard C/C++ compilers. 



Run this command to install essential compilation tools: 



&#x20;sudo apt install build-essential libssl-dev libffi-dev python3-dev -y



Expected Output: 



Reading package lists... Done

Building dependency tree... Done

The following ADDITIONAL packages will be installed:

&#x20; g++, gcc, make, libc6-dev...

0 upgraded, 12 newly installed, 0 to remove.



3\. Language-Specific Dependency Managers



To manage project dependencies clearly, always use the dedicated package managerfor your runtime environment. 



Python Dependencies (pip and requirements.txt) 



Instead of installing Pythin dependencies globally, use pip inside your activated virtual environment to isolate project packages:

&#x20;# 1. Create a file containing your project dependencies

nano requirements.txt



Type your required libraries inside the editor (e.g. requests==2.31.0), then press Ctrl+0, Enter and Ctrl+X to save and exit. 



Command:

\# 2. Install all dependencies listed in the file

pip install -r requirements.txt



Expected Output: 

Collecting requests==2.31.0 (from -r requirements.txt)

&#x20; Downloading requests-2.31.0-py3-none-any.whl (62 kB)

Installing collected packages: requests

Successfully installed requests-2.31.0



Node.js Dependencies (npm and package.json)



Node.js relies on NPM to fetch and track packages inside a local node\_modules directory



Command:

\# 1. Initialize a new project configuration file

npm init -y



\# 2. Install a sample package (e.g., Express framework)

npm install express



Expected Output: 

added 64 packages, and audited 65 packages in 2s

found 0 vulnerabilities



(This creates a package.json file and a  node\_module folder inside your current directory). 



4\. Verifying Installed Dependencies 



You can audit your environment at anytime to verify that all required dependencies are present and match the correct versions. 

* For Python: Run pip list or pip freeze to see all active environment packages. 
* For Node.js: Run npm list --depth=0 to view top-level project dependencies. 



&#x20;

