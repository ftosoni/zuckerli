import struct, sys

def get_data(infilepath) :
    infile = open(infilepath, 'r')
    line = infile.readline()
    yield int(line[:-1])
    line = infile.readline()
    while line :
        assert(line[-1] == '\n')
        yield [int(x) for x in line[:-1].split()]
        line = infile.readline()
    infile.close()

def main() :  
    if len(sys.argv) != 2+1 :
        print('Usage is:', sys.argv[0], '<infilepath> <outfilepath>') 
        exit(-1)

    #args
    infilepath = sys.argv[1]
    outfilepath = sys.argv[2]

    #business logic from here
    outfile = open(outfilepath, 'wb')

    #fingerprint
    b = struct.pack('<I', 132) #8) #bytes nedges
    outfile.write(b)
    b = struct.pack('<I', 0) #4) #bytes nnodes
    outfile.write(b)

    #ps
    for i,adjl in enumerate(get_data(infilepath)) :
        if i==0 :
            #nnodes
            nnodes = adjl
            b = struct.pack('<I', nnodes)
            outfile.write(b)

            #first acc
            acc = 0
            b = struct.pack('<Q', acc)
            outfile.write(b)

        else :
            acc += len(adjl)
            b = struct.pack('<Q', acc)
            outfile.write(b)

    #edges
    for i,adjl in enumerate(get_data(infilepath)) :
        if i==0 :
            continue
        for e in adjl :
            b = struct.pack('<I', e)
            outfile.write(b)
    outfile.close()
        
        
if __name__ == '__main__' :
    main()