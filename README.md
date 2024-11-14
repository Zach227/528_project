# High Performance Computing (HPC) Lab Repository

To complete the labs for the course, you will need to install the HPC Lab repository. Download and extract the following archive file to a work area in your home directory.

hpc-labs.tgz

```
tar -xzf hpc-labs.tgz
```

## Compiling and Running Labs

Several common build related commands are available with the "make" utility. From the project directory at the command prompt:

- Type "make" to build the lab.
- Type "make run" to run the lab
- Type "make clean" to delete build files
- Type "make submit" to create a zip file for submission

## Using Lab Template Code

Within each lab are "template" files to jump start your lab coding assignment. Template files are meant to be used as starting code, whereas solution files are meant for your implementation. Copy any "template" source files to new files named "solution" with the same extension. For example, copy template.cu to solution.cu and then implement your lab in solution.cu (leaving template.cu untouched). Look for comments in the code starting with //@@ and add the code as directed just below the comment.

## Other Helpful Configuration

From the VSCode menu, select Run -> Install Additional Debuggers...

- Install C/C++ for Visual Studio Code.
- Install Nsight Visual Studio Code Edition.
- Install Makefile Tools.

Note: You can turn off telemetry in settings (gear icon).
# 528_project
