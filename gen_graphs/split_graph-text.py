import sys

def get_data(infilepath) :
    infile = open(infilepath, 'r')
    line = infile.readline()
    while line :
        yield line
        line = infile.readline()
    infile.close()

def split(infilepath, blocks) :
    ext = '.graph-text'
    assert(infilepath[-len(ext):] == ext)

    tid = 0
    to_read = 0
    outfile = None
    for i,line in enumerate(get_data(infilepath)) :
        if i==0 :
            #general
            rows = int(line)
            row_block_size = (rows + blocks - 1) // blocks  
            continue          
        if not to_read :
            if outfile is not None :
                #write empty lines
                while (outfile.row < rows) :
                    outfile.write('\n')
                    outfile.row += 1
                #close
                assert(outfile.row == rows)
                outfile.close()
            outfilepath = f'{infilepath[:-len(ext)]}.{blocks}.{tid}.graph-text' 
            outfile = open(outfilepath, 'w')
            outfile.row = 0
            assert(tid <= blocks)
            outfile.write(f'{rows}\n')
            for _ in range(tid*row_block_size) :
                outfile.write('\n')
                outfile.row += 1
            outfile.write(line)
            outfile.row += 1
            to_read = row_block_size - 1
            tid += 1
        else :
            outfile.write(line)
            outfile.row += 1
            to_read -= 1
    #print('tid:', tid)
    assert(tid == blocks)
    if outfile is not None :
        #write empty lines
        while (outfile.row < rows) :
            outfile.write('\n')
            outfile.row += 1
        #close
        assert(outfile.row == rows)
        outfile.close()

def main() :
    if len(sys.argv) != 2+1 :
        print('Usage is:', sys.argv[0], '<infilepath> <par. degree>')
        exit(-1)
    
    #args
    infilepath = sys.argv[1]
    blocks = int(sys.argv[2])

    split(infilepath, blocks)

if __name__ == '__main__' :
    main()
