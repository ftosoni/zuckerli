# Zuckerli
Compression format and data structures for large graphs.

## Cloning and compiling

``` shell
git clone https://github.com/google/zuckerli
cd zuckerli
git submodule init
git submodule update
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
