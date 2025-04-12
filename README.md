# Zuckerli
Compression format and data structures for large graphs.

## Matrix Multiplication and PageRank  

If you employ the matrix-vector multiplication algorithm, please cite the following paper:  

> F. Tosoni, P. Bille, V. Brunacci, A. d. Angelis, P. Ferragina and G. Manzini, "Toward Greener Matrix Operations by Lossless Compressed Formats," in IEEE Access, vol. 13, pp. 56756-56779, 2025, doi: [10.1109/ACCESS.2025.3555119](https://doi.org/10.1109/ACCESS.2025.3555119). keywords: {Sparse matrices;Energy consumption;Energy efficiency;Software;Vectors;Servers;Internet of Things;Big Data;Meters;Machine learning algorithms;Green computing;lossless compression techniques;compressed matrix formats;matrix-to-vector multiplications;computation-friendly compression;PageRank}, 


```tex
@ARTICLE{tosoni2024greenermatrixoperationslossless,
  author={Tosoni, Francesco and Bille, Philip and Brunacci, Valerio and Angelis, Alessio de and Ferragina, Paolo and Manzini, Giovanni},
  journal={IEEE Access}, 
  title={Toward Greener Matrix Operations by Lossless Compressed Formats}, 
  year={2025},
  volume={13},
  number={},
  pages={56756--56779},
  keywords={Sparse matrices;Energy consumption;Energy efficiency;Software;Vectors;Servers;Internet of Things;Big Data;Meters;Machine learning algorithms;Green computing;lossless compression techniques;compressed matrix formats;matrix-to-vector multiplications;computation-friendly compression;PageRank},
  doi={10.1109/ACCESS.2025.3555119}}
```  

**Credits**: The Zuckerli compressed matrix format is attributed to the authors of the original repository: [https://github.com/google/zuckerli](https://github.com/google/zuckerli) and its associated sources. 

## Cloning and compiling

``` shell
git clone https://github.com/google/zuckerli
cd zuckerli
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 12
```

Note that to compile and run tests `googletest` should be installed on your
system (and apparently `clang` too).

``` shell
 sudo apt install clang libgtest-dev libgmock-dev
```
## Running
### Encoding

``` shell
./encoder --input_path example --output_path example.zkr
```

### Decoding
``` shell
./decoder --input_path example.zkr
```

### Multiplying
``` shell
./multiplier --input_path example.zkr --input_vector_path invec14
```
