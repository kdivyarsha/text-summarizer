import csv
sfilename="/Users/thejashreekunam/Documents/summarizer-master/Reviews.csv"

def getuniqueprodids(sfilename):
    prodid=[]
    with open(sfilename,encoding="UTF-8",mode="r" ,newline="") as csv_file:
        csreader = csv.reader(csv_file)
        i=0
        for x in csreader:
            if(i==0):
                i+=1
                continue
            prodid.append(x[1])
    prodset=set(prodid)
    produniquelist=list(prodset)
    return produniquelist   

prodlist=getuniqueprodids(sfilename)
for item in prodlist:
    with open("/Users/thejashreekunam/Documents/summarizer-master/files/{}.txt".format(item),"w+") as csv_infile:
        cswriter=csv.writer(csv_infile,delimiter="\t")
        with open(sfilename,encoding="UTF-8",mode="r" ,newline="") as csv_file:
            csreader = csv.reader(csv_file)
            for x in csreader:
                if(x[1]==item):
                    cswriter.writerow([x[2],x[6],x[8],x[4],x[5],x[9]])
