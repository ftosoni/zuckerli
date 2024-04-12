import struct, sys

def main() :
    if len(sys.argv) != 1+1 :
        print('Usage is:', sys.argv[0], '<#nodes>')
        exit(-1)
    
    #args
    nnodes = int(sys.argv[1])

    outfilepath = f'invec{nnodes}'
    outfile = open(outfilepath, 'wb')

    for i in range(nnodes) :
        b = struct.pack('d', i%10)
        outfile.write(b)

    outfile.close()

if __name__ == '__main__' :
    main()