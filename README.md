## An Open-source, Single-GPU, Large-scale Thermal Microstructure Design Framework

This project aims to provide an code framework for efficiently solving the inverse homogenization problems to design thermal microstructure.  

### dependency

* OpenVDB
* CUDA11.6+
* glm
* Eigen3
* Anaconda
* Visual Studio 2022

For CUDA's version and Visual Studio's version, we tested on several different machines, and using this combination minimizes the probability of unknown errors, ensuring the highest possible stability in operation. We have packed dependencies into a [conda](https://docs.conda.io/en/latest/miniconda.html) environment (except CUDA  and compilers), you can create it by:

```bash
conda env create -f environment.yml
```

Then you activate it by:

```bash
conda activate openTM
```
##### Tips

If you have previously used vcpkg to configure your C++ environment, please modify
```bash
set(CMAKE_IGNORE_PATH "C:/vcppkg/vcpkg/installed/x64-windows/share/gflags")
```
in the CMakeLists to avoid conflicts.


### Compilation

After the dependency is installed, the code can be compiled using cmake:

```shell
mkdir build
cd build
cmake ..
```

If the conda environment is activated, `cmake` will automatically checkout the dependencies in this environment.

After opening the .sln file, go to Project -> Properties -> Configuration Properties -> C/C++ -> Command Line, and enter \bigobj.
In Project -> Properties -> Configuration Properties -> Debugging -> Environment, add PATH=(your conda path).conda\envs\homo3d\Library\bin;$(PATH) or you can just drag these .dll by hand, and also place the compiled openvdb.dll into the build directory.

### Usage

Open the .sln in build folder and set homo3d as the startup project. Thus you are running the example in main.

After the optimization finished, the optimized density field is stored in `<prefix>/rho` in OpenVDB format.

3rd party softwares like Rhino (with grasshopper plugin [Dendro](https://www.food4rhino.com/en/app/dendro)) or Blender may be used to extract the solid part.

### Compile the Python interface.
You need to add the pybind11 and anaconda support to compile the uncommented code in main function. 

See the video as example.
We have changed our project name from homo3d to openTM. Use 
```python
import openTM
help(openTM)
```
instead.
