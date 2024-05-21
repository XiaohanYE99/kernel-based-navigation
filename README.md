

# LibRVO

![](example.gif)

This library implements our differentiable, stable algorithm for multi-agent navigation. The library supports the following features:
- Optimization-based RVO algorithm that avoids jittering of agents
- Acceleration spatial hash structure for neighbor search
- Agents with variable radius
- Differentiable interface for model-based machine learning
- Multi-threaded batched RVO for machine learning
- Python interface

## Install Instructions

The library uses CMake as its build tool. The basic RVO library can be run without external dependence. However, the following additional features require external libraries:
- Cholmod for faster linear system solve (download and install this library from: https://people.engr.tamu.edu/davis/suitesparse.html)
- TinyVisualizer for OpenGL visualization (download and install this small library from: https://github.com/gaoxifeng/TinyVisualizer.git)
- Boost/MPFR/GMP for multi-precision support (these are currently used for debugging and we suggest against actually using them during runtime)

### Windows Installation
On windows, we recommend using vcpkg to install all the dependencies via the following commands:
```
cd /d C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install eigen3
vcpkg install glfw3:x64-windows
vcpkg install boost:x64-windows-static
```
After install all dependencies, we can move forward to install TinyVisualizer:
```
cd /d C:\
git clone https://github.com/gaoxifeng/TinyVisualizer.git
cd TinyVisualizer
git submodule update --recursive --init
mkdir C:\TinyVisualizer-build
cd C:\TinyVisualizer-build
cmake C:\TinyVisualizer -DCMAKE_BUILD_TYPE=Debug -DEIGEN3_INCLUDE_DIR=C:\vcpkg\installed\x64-windows\include\eigen3 -DGLFW_INCLUDE_DIR=C:\vcpkg\installed\x64-windows\include -DGLFW_LIBRARIES=C:\vcpkg\installed\x64-windows\lib\glfw3dll.lib
```
Then you could open: ```C:\TinyVisualizer-build\TinyVisualizer.sln``` and run the ```INSTALL``` to install the debug version. If you would like to install the release version, change the above option to: ```-DCMAKE_BUILD_TYPE=Release```. After install all dependencies, we can install the ```RVO``` library itself:
```
cd /d C:\
git clone https://github.com/XiaohanYE99/kernel-based-navigation.git
cd C:/kernel-based-navigation
git checkout variable-radius-rvo
git submodule update --recursive --init
mkdir C:\kernel-based-navigation-build
cd C:\kernel-based-navigation-build
cmake C:\kernel-based-navigation -DCMAKE_BUILD_TYPE=Debug -DEIGEN3_INCLUDE_DIR=C:\vcpkg\installed\x64-windows\include\eigen3 -DGLFW_INCLUDE_DIR=C:\vcpkg\installed\x64-windows\include -DGLFW_LIBRARIES=C:\vcpkg\installed\x64-windows\lib\glfw3dll.lib -DBoost_INCLUDE_DIR=C:\vcpkg\installed\x64-windows-static\include
```
Now you can open ```C:\kernel-based-navigation-build``` and play with it. Note that the ```CMAKE_BUILD_TYPE``` should match in two projects. Also, we assume cmake is installed and added to system path, otherwise, one could use the GUI interface of cmake.

### Ubuntu Installation
Installation on ubuntu is much easier and I assume the installation is under home direction, then we start by preparing the packages:
```
sudo apt install libglfw3 libboost-dev libeigen3-dev
cd ~
git clone https://github.com/gaoxifeng/TinyVisualizer.git
cd ~/TinyVisualizer
git submodule update --recursive --init
mkdir ~/TinyVisualizer-build
cd ~/TinyVisualizer-build
cmake ../TinyVisualizer -DCMAKE_BUILD_TYPE=Debug
sudo make install
```
And then we move on to install ```RVO```:
```
cd ~
git clone https://github.com/XiaohanYE99/kernel-based-navigation.git
cd ~/kernel-based-navigation
git checkout variable-radius-rvo
git submodule update --recursive --init
mkdir ~/kernel-based-navigation-build
cd ~/kernel-based-navigation-build
cmake ../kernel-based-navigation -DCMAKE_BUILD_TYPE=Debug
sudo make install
```

## Python Binding
If you would like to call RVO from python, you could add the the option ```-DPYTHON_BINDING=Python3``` to the cmake command, and a new library will be compiled for python binding.

## Examples

We provide the following example programs that are self-explanatory:
|Example          |Explanation                                                  |
|----------------------|--------------------------------------------------------|
|mainSimulator.cpp     |C++ example of single environment                       |
|mainMultiSimulator.cpp|C++ example of batched multiple environments            |
|testRVO.py            |Python3 example of single environment                   |
|testMultiRVO.py       |Python3 example of batched multiple environments        |
