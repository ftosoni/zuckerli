import os, sys

def main() :
    if len(sys.argv) != 2+1 :
        print('Usage is:', sys.argv[0], '<graph-text> <par. degree>')
        exit(-1)
    
    #args
    infilepath = sys.argv[1]
    pardegree = int(sys.argv[2])

    #params
    builddir = '../cmake-build-debug'
    ext = '.graph-text'
    basename = infilepath[:-len(ext)]

    cmd = f'python3 split_graph-text.py {infilepath} {pardegree}'
    os.system(cmd)

    for tid in range(pardegree) :
        cmd = f'python3 graph-text2zkr-plain.py \
              {basename}.{pardegree}.{tid}.graph-text \
              {basename}.{pardegree}.{tid}.zkr-plain '
        os.system(cmd)

        cmd = f'{builddir}/encoder \
            --input_path {basename}.{pardegree}.{tid}.zkr-plain \
            --output_path {basename}.{pardegree}.{tid}.zkr'
        os.system(cmd)

if __name__ == '__main__' :
    main()

    