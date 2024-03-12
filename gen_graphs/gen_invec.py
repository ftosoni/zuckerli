import struct

outfilepath = 'invec14'
outfile = open(outfilepath, 'wb')

for i in range(14) :
    b = struct.pack('d', i%10)
    outfile.write(b)

outfile.close()