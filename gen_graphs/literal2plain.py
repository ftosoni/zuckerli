import struct

def get_data(infilepath) :
    infile = open(infilepath, 'r')
    line = infile.readline()
    while line :
        assert(line[-1] == '\n')
        yield [int(x) for x in line[:-1].split()]
        line = infile.readline()
    infile.close()

def num_lines(infilepath) :
    infile = open(infilepath, 'r')
    n = 0
    line = infile.readline()
    while line :
        n += 1
        line = infile.readline()
    infile.close()
    return n

if __name__ == '__main__' :
    infilepath = 'example14.txt'
    outfilepath = 'example14'
    
    outfile = open(outfilepath, 'wb')

    #fingerprint
    b = struct.pack('<I', 132) #8) #bytes nedges
    outfile.write(b)
    b = struct.pack('<I', 0) #4) #bytes nnodes
    outfile.write(b)

    #nnodes
    nnodes = num_lines(infilepath)
    b = struct.pack('<I', nnodes)
    outfile.write(b)

    #ps
    acc = 0
    b = struct.pack('<Q', acc)
    outfile.write(b)
    for adjl in get_data(infilepath) :
        acc += len(adjl)
        b = struct.pack('<Q', acc)
        outfile.write(b)

    #edges
    for adjl in get_data(infilepath) :
        for e in adjl :
            b = struct.pack('<I', e)
            outfile.write(b)
    outfile.close()
        
        
