f=open("file_dataset/7rettetest.txt","r")
o=open("file_dataset/7rettetest1.txt","w")

for line in f.readlines():
    if line != "inf\n":
        o.write(line)
    else:
        o.write("{:}\n".format(-1))