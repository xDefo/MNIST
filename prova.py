f=open("prova.txt","r")

for lines in f:
    
    print(int(lines[0:a]))
    print(int(lines[a:(len(lines))]))