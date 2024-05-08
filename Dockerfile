# Base Image Selection
FROM python:3.9-slim-buster  
# "base image" refers to the initial image upon which you build your own custom image.
# This base image contains the Python 3.9 runtime environment installed on a slim version of the Debian Buster operating system.

# python:3.9-slim-buster as it's a minimal Python image based on Debian Buster, providing a balance between size and compatibility.
# Debian is a free and open-source Linux distribution known for its stability, security, and extensive package repositories.
# "open-source Linux distribution" refers to a version of the Linux operating system that is built using open-source software

# compatibility in the context of Docker image selection ensures that the chosen base image provides the necessary environment 
# and resources for the application to run correctly without encountering compatibility issues or conflicts with its dependencies.
# if the application relies on Linux-specific features or libraries, it should be built on a Linux-based image
# The base image should support all the software dependencies and platform requirements of the application
# Software dependencies are other software components or libraries that a particular application relies on in order to function properly
# if the application is written in Python, the base image should include a compatible version of the Python interpreter.

# When choosing a specific version tag 
# reliability means that the system operates consistently and performs as expected under normal conditions, without unexpected failures or errors like no errors when update the image .
# By using the specific version tag (e.g., "2.4.41") instead of the "latest" tag in their Dockerfile, the developers ensure predictability because may be the latest version
# has new features and secrutiy patches 
# stability consider predictabily and compitability


# The image provides only the most necessary components required for running Python applications.
# It doesn't include unnecessary extras, which reduces the potential points of vulnerability or attack.
# Software packages often contain bugs or flaws that could be exploited by attackers. 
# These vulnerabilities may allow attackers to gain unauthorized access to the system, execute arbitrary code, or perform other malicious actions.


# second reqirement in the coversheet 
# Docker official images, like python:3.9-slim-buster, are regularly maintained and updated by the Docker community and maintainers. 
# This means that security vulnerabilities discovered in the base image or its dependencies are promptly patched and fixed



# Set working directory
WORKDIR / C:\dockerize\Project\project 2
# sets the working directory for any subsequent instructions in the Dockerfile
# WORKDIR is used to set the working directory to /app, avoiding the need for multiple RUN cd commands.

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean
# RUN: This instruction allows you to execute commands inside the Docker container during the image build process.
# 'apt-get update`: This command  updates the package index of the package manager (`apt-get`). It ensures that the package manager knows about the latest versions of packages available in the repositories.
# A package manager is a software tool used to automate the process of installing, upgrading, configuring, and removing software packages on a computer system.
#  It simplifies the management of software dependencies by providing a centralized repository of packages that can be easily installed and managed.
# package managers play a crucial role in software management by simplifying installation, upgrading, and removal of software packages, 
# ensuring compatibility between different software components, and contributing to system stability and reliability.
#  Package managers simplify the process of installing software by handling dependencies automatically. When you install a package, 
# the package manager checks for and installs any other packages (dependencies) required by the software you're installing.


# apt-get update: This command updates the local package index, which is a database of available packages and their versions, 
# from the repositories configured in the system. It ensures that the package manager has the latest information about available packages and their dependencies.

# &&: This is a command chaining operator in Linux, which allows you to execute multiple commands sequentially. 
# In this case, it ensures that the apt-get update command is successfully executed before proceeding to the next command.

# apt-get install -y libgl1-mesa-glx libglib2.0-0: This command installs the specified system dependencies using the package manager (apt-get). 
# The -y flag automatically answers "yes" to any prompts asking for confirmation during the installation process. 
# The packages being installed are libgl1-mesa-glx and libglib2.0-0, which are system libraries required for OpenCV, a computer vision library. 
# These dependencies are necessary for the proper functioning of OpenCV-related applications within the Docker container.



# Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .
# By copying the requirements.txt file before installing dependencies, Docker can leverage its caching mechanism. 
# If the requirements file hasn't changed since the last build, Docker will reuse the cached layer containing the requirements installation step,
# resulting in faster builds and reduced disk usage.


# In Docker, each instruction in a Dockerfile creates a layer in the image. When Docker builds an image, 
# it caches these layers to improve build performance and reduce redundant work.

# When a Dockerfile is executed, Docker checks each instruction to determine whether it can be served from the cache or if it needs to be rebuilt. 
# If an instruction is identical to one that has been executed before and the context 
# (e.g., files in the build directory) hasn't changed, Docker can reuse the previously cached layer instead of rebuilding it from scratch.

# In the context of the provided Dockerfile snippet, copying the requirements.txt file before installing dependencies is a strategy
#  to leverage Docker's caching mechanism. If the requirements.txt file hasn't changed since the last build,
# Docker will reuse the cached layer containing the requirements installation step. 
# This means that subsequent builds will be faster because Docker won't need to reinstall the dependencies if they haven't changed, 
# leading to improved build performance and reduced disk usage.


# Checking for Cached Layers: When Docker encounters an instruction during a build, it checks if there is a cached layer corresponding to that instruction. 
# It does this by comparing the instruction and its context (e.g., files in the build directory) with the previously built layers.


# # Install dependencies (Python packages)
# RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt
# default-timeout=100: This option sets the maximum time in seconds that pip will wait for a package to download before timing out. In this case, it sets the timeout to 100 seconds. 
# This can be useful when downloading packages from remote repositories, ensuring that pip has sufficient time to download larger packages or when the network connection is slow.

# Install dependencies (Python packages)
RUN pip install --no-cache-dir -r requirements.txt

# By default, pip install caches downloaded package files in a cache directory (~/.cache/pip by default). When you use pip install without the --no-cache-dir flag,
# Docker will cache these downloaded package files in intermediate image layers during the build process. 
# This can increase the size of the Docker image layers and the overall size of the final Docker image.


# -r requirements.txt: This specifies a requirement file (requirements.txt) that contains a list of Python packages and their versions

# Copy the rest of the application code
COPY . .
# This line copies all the files and directories from the current directory on the host machine (where the Dockerfile is located) into the root directory (/) of the Docker container the root directory / of the Docker image being built..
# .: This specifies the source path on the host machine. In this case, . represents the current directory, which typically contains all the application code and files needed for the Docker image.

# Docker images need to contain all the files and resources required to run your application. By copying files from the host machine into the Docker image, 
# you ensure that the container has access to the application code, configuration files, scripts, and any other necessary assets.

# using copy instead of add This helps to maintain clarity and predictability in your Dockerfile, as COPY is more explicit about what it does.
# add -> URL Support and Automatic Extraction: If the source file is a tar archive (commonly ending with .tar.gz, .tar, .tgz, .tbz2, .txz), ADD will automatically extract it into the destination directory in the image.

# Set the command to run the application
CMD ["python", "cloud.py"]

# CMD: This instruction specifies the default command to run when the container starts.
# ["python", "cloud.py"]: This specifies the command to run inside the container. In this case, 
# it runs a Python script named cloud.py. The python command is used to execute Python scripts.