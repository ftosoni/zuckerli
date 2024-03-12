import os 
import sys

def read_graph(infilepath):
    with open(infilepath, 'rb') as file:
        bytes_nedges_b = file.read(4)
        bytes_nedges = int.from_bytes(bytes_nedges_b, byteorder='little')

        bytes_nnodes_b = file.read(4)
        bytes_nnodes = int.from_bytes(bytes_nnodes_b, byteorder='little')

        nnodes_b = file.read(4)
        nnodes = int.from_bytes(nnodes_b, byteorder='little')

        ps = [int.from_bytes(file.read(8), byteorder='little') for _ in range(nnodes+1)]

        ds = [int.from_bytes(file.read(4), byteorder='little') for _ in range(ps[-1])]

    return bytes_nedges, bytes_nnodes, nnodes, ps, ds

if len(sys.argv) != 1+1:
    print("Usage is:", sys.argv[0], '<infilepath>')
    sys.exit(1)

#args
infilepath = sys.argv[1]

bytes_nedges, bytes_nnodes, nnodes, ps, ds = read_graph(infilepath)

print("bytes nedges:", bytes_nedges)
print("bytes nnodes:", bytes_nnodes)
print("nnodes:", nnodes)    
print("ps:", ps)
assert(nnodes+1 == len(ps))
for i in range(nnodes) : 
    fst = ps[i]
    lst = ps[i+1]
    adj = ds[fst:lst]
    
    prev = -1
    print(f'{i}: {adj}')
print('est. dim.:', 20 + 4*(2*nnodes + ps[-1]))
print('dim.:', os.path.getsize(infilepath))