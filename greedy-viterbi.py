


import numpy as np




trainData = open('data/train', 'r')
trainData_lines = trainData.readlines()




wordList_train = []
for line in trainData_lines:
    wordList_train.append(line.split())




wordCountDict = {}




for subTestList in wordList_train:
    if subTestList != []:
        if subTestList[1] in wordCountDict:
            wordCountDict[subTestList[1]]+=1
        else:
            wordCountDict[subTestList[1]] = 1





wordCountDict_new = {}





unkWords = set()
nonUnkWords = set()





for k,v in wordCountDict.items():
    if "<unk>" not in wordCountDict_new:
        wordCountDict_new["<unk>"] = 0
        wordCountDict_new[k] = v
        nonUnkWords.add(k)
        nonUnkWords.add("<unk>")
    else:
        if v <= 3:
            unkWords.add(k)
            wordCountDict_new["<unk>"]+=v
        else:
            nonUnkWords.add(k)
            wordCountDict_new[k] = v





wordCountDict_new = dict(sorted(wordCountDict_new.items(), key = lambda x: x[1], reverse = True)) ####change





vocabFile = open('vocab.txt', 'w')

c = 0

vocabFile.write("<unk>")
vocabFile.write("\t")
vocabFile.write(str(c))
vocabFile.write("\t")
vocabFile.write(str(wordCountDict_new["<unk>"]))
vocabFile.write("\n")
c+=1

del wordCountDict_new["<unk>"]

for k,v in wordCountDict_new.items():
    vocabFile.write(k)
    vocabFile.write("\t")
    vocabFile.write(str(c))
    vocabFile.write("\t")
    vocabFile.write(str(v))
    vocabFile.write("\n")
    c+=1

vocabFile.close()





posTagCountDict = {}





for subWordList in wordList_train:
    if subWordList != []:
        if subWordList[2] in posTagCountDict:
            posTagCountDict[subWordList[2]]+=1
        else:
            posTagCountDict[subWordList[2]] = 1





transitionVocab_numerator_withSTART = {}





for i in range(len(wordList_train)):
    if wordList_train[i] != []:
        if wordList_train[i][0] != "1":
            prevTag = wordList_train[i-1][2]
            curTag = wordList_train[i][2]
            if (prevTag,curTag) in transitionVocab_numerator_withSTART:
                transitionVocab_numerator_withSTART[(prevTag,curTag)]+=1
            else:
                transitionVocab_numerator_withSTART[(prevTag,curTag)] = 1
        else:
            prevTag = "START"
            curTag = wordList_train[i][2]
            if (prevTag,curTag) in transitionVocab_numerator_withSTART:
                transitionVocab_numerator_withSTART[(prevTag,curTag)]+=1
            else:
                transitionVocab_numerator_withSTART[(prevTag,curTag)] = 1
            
            if prevTag in posTagCountDict:
                posTagCountDict[prevTag]+=1
            else:
                posTagCountDict[prevTag] = 1





transitionVocab_probab_withSTART = {}
transitionVocab_numerator_withSTART_tup = {}





for k,v in transitionVocab_numerator_withSTART.items():
    transitionVocab_numerator_withSTART_tup[k] = transitionVocab_numerator_withSTART[k]/posTagCountDict[k[0]]
    transitionVocab_probab_withSTART[str(k)] = transitionVocab_numerator_withSTART[k]/posTagCountDict[k[0]]





del posTagCountDict['START']





for w in wordList_train:
    if w != []:
        if w[1] in unkWords:
            w[1] = "<unk>"





emissionVocab_numerator = {}





for subWordList in wordList_train:
    if subWordList != []:
        if (subWordList[2],subWordList[1]) in emissionVocab_numerator:
            emissionVocab_numerator[(subWordList[2],subWordList[1])]+=1
        else:
            emissionVocab_numerator[(subWordList[2],subWordList[1])] = 1





for word in nonUnkWords:
    for k in posTagCountDict.keys():
        if (k,word) not in emissionVocab_numerator:
            emissionVocab_numerator[(k,word)] = 0





emissionVocab_probab = {}
emissionVocab_probab_tup = {}





for k in emissionVocab_numerator.keys():
    if emissionVocab_numerator[k] == 0:
        emissionVocab_probab_tup[k] = 0
        emissionVocab_probab[str(k)] = 0
    else:
        emissionVocab_probab_tup[k] = emissionVocab_numerator[k]/posTagCountDict[k[0]]
        emissionVocab_probab[str(k)] = emissionVocab_numerator[k]/posTagCountDict[k[0]]





import json





transition_emission_dict = {'Transition':transitionVocab_probab_withSTART, 'Emission':emissionVocab_probab}





with open("hmm.json", "w") as hmmFile:
    json.dump(transition_emission_dict, hmmFile, indent=4)





devData = open('data/dev', 'r')
devData_lines = devData.readlines()

wordList_dev = []
devCopy = []





for line in devData_lines:
    wordList_dev.append(line.split())











for line in devData_lines:
    devCopy.append(line.split())





listOfTags = list(posTagCountDict.keys())





for w in devCopy:
    if w != []:
        if w[1] in unkWords:
            w[1] = "<unk>"





devWords = []





for w in wordList_dev:
    if w != []:
        devWords.append(w[1])





probabDict = {}
#wordPredictDict = {}
wordPredictList = []





for i in range(len(wordList_dev)):
    if wordList_dev[i] != []:
        for j in range(len(listOfTags)):
            if wordList_dev[i][0]== "1":
                prevTag="START"
            curTag = listOfTags[j]
            if (curTag,devCopy[i][1]) not in emissionVocab_probab_tup:
                ep = emissionVocab_probab_tup[(curTag,"<unk>")]
            else:
                ep = emissionVocab_probab_tup[(curTag,devCopy[i][1])]
            if (prevTag,curTag) not in transitionVocab_numerator_withSTART_tup:
                tp = 0
            else:
                tp = transitionVocab_numerator_withSTART_tup[(prevTag,curTag)]
                
            probabDict[curTag] = tp*ep
        maxTag = max(probabDict, key = probabDict.get)
        #wordPredictDict[wordList_dev[i][1]] = maxTag
        wordPredictList.append([wordList_dev[i][1],maxTag])
        prevTag=maxTag





blank = wordList_dev.count([])





for i in range(blank):
    wordList_dev.remove([])
    devCopy.remove([])





# noOfCorrectPredictions = 0
c=0
for i in range(len(wordList_dev)):
        #if wordList_dev[i][2]==wordPredictDict[wordList_dev[i][1]]:
        if wordList_dev[i][2]==wordPredictList[i][1]:
            c+=1     





print("The accuracy of greedy decoding algorithm (dev file)",c/131768)







tstData = open('data/test', 'r')
tstData_lines = tstData.readlines()

wordList_tst = []
tstCopy = []





for line in tstData_lines:
    wordList_tst.append(line.split())
    tstCopy.append(line.split())





for w in tstCopy:
    if w != []:
        if w[1] in unkWords:
            w[1] = "<unk>"





probabDict_tst = {}
wordPredictList_tst = []





for i in range(len(wordList_tst)):
    if wordList_tst[i] != []:
        for j in range(len(listOfTags)):
            if wordList_tst[i][0]== "1":
                prevTag="START"
            curTag = listOfTags[j]
            if (curTag,tstCopy[i][1]) not in emissionVocab_probab_tup:
                ep = emissionVocab_probab_tup[(curTag,"<unk>")]
            else:
                ep = emissionVocab_probab_tup[(curTag,tstCopy[i][1])]
            if (prevTag,curTag) not in transitionVocab_numerator_withSTART_tup:
                tp = 0
            else:
                tp = transitionVocab_numerator_withSTART_tup[(prevTag,curTag)]
                
            probabDict_tst[curTag] = tp*ep
        maxTag = max(probabDict_tst, key = probabDict_tst.get)
        #wordPredictDict[wordList_dev[i][1]] = maxTag
        wordPredictList_tst.append([wordList_tst[i][0],wordList_tst[i][1],maxTag])
        prevTag=maxTag





wordPredictList_tst[len(wordPredictList_tst)-2]





with open('greedy.out', 'w') as greedyFile:

    for i in range(len(wordPredictList_tst)):
        if i+1 != len(wordPredictList_tst) and wordPredictList_tst[i+1][0] == "1":
            greedyFile.write(wordPredictList_tst[i][0])
            greedyFile.write("\t")
            greedyFile.write(wordPredictList_tst[i][1])
            greedyFile.write("\t")
            greedyFile.write(wordPredictList_tst[i][2])
            
            
            greedyFile.write("\n")
            
            greedyFile.write("\n")
            
        else:
            c+=1
            greedyFile.write(wordPredictList_tst[i][0])
            greedyFile.write("\t")
            greedyFile.write(wordPredictList_tst[i][1])
            greedyFile.write("\t")
            greedyFile.write(wordPredictList_tst[i][2])
            greedyFile.write("\n")
            

greedyFile.close()











def getTP(pt,ct):
    if (pt,ct) not in transitionVocab_numerator_withSTART_tup:
        tp = 0
    else:
        tp = transitionVocab_numerator_withSTART_tup[(pt,ct)]
    
    return tp





def getEP(ct,w):
    if (ct,w) not in emissionVocab_probab_tup or w in unkWords:
        ep = emissionVocab_probab_tup[(ct,"<unk>")]
    else:
        ep = emissionVocab_probab_tup[(ct,w)]
    
    return ep





import operator





def viterbi(sent):
    sent = sent.split("\n")
    
    probabDict_vd = np.zeros((len(listOfTags),len(sent))).astype("float64")
    bckProp = np.zeros((len(listOfTags),len(sent))).astype("float64")
    c = np.zeros((len(listOfTags),len(sent))).astype("float64")
    opt=np.zeros((len(sent)))
    
    for i in range(len(sent)):
        words = sent[i].split("\t")
        for ct in range(len(listOfTags)):
            if words[0] == "1":
                trans_ct_f = []
                for pt in range(len(listOfTags)):
                    tp = getTP("START",listOfTags[ct])
                    ep = getEP(listOfTags[ct],words[1])

                    trans_ct_f.append(tp*ep)

                probabDict_vd[ct,i] = max(trans_ct_f)
            else:
                trans_ct = []
                for pt in range(len(listOfTags)):
                    trans_ct.append(getTP(listOfTags[pt], listOfTags[ct]))
                c = probabDict_vd[:, i-1]*trans_ct

                bckProp[ct,i],probabDict_vd[ct,i] = max(enumerate(c),key=operator.itemgetter(1))

                ep = getEP(listOfTags[ct],words[1])

                probabDict_vd[ct,i] = probabDict_vd[ct,i]*ep

    opt[len(sent)-1] = probabDict_vd[:,len(sent)-1].argmax()
    for j in range(len(sent)-1,0,-1):
        opt[j-1] = bckProp[int(opt[j]),j]
    
    s = 0
    for k in range(len(sent)):
        word = sent[k].split("\t")
        if len(word) < 3:
            s = 0
        elif listOfTags[int(opt[k])]==word[2]:
            s+=1
    return s,opt





devData_v = open('data/dev', 'r').read().strip()
devData_sent = devData_v.split("\n\n")

sum = 0
pathLst = []
for i in range(len(devData_sent)):
    val,path = viterbi(devData_sent[i])
    pathLst.append(path)
    sum = sum + val

print("The accuracy of viterbi decoding algorithm (dev file)",sum/131768)














sentWords = []
for i in range(len(devData_sent)):
    sent = devData_sent[i].split("\n")
    for j in range(len(sent)):
        s = sent[j].split()
        sentWords.append(s)
        #print(s[1])
        #print(pathLst[i][j])





testData_v = open('data/test', 'r').read().strip()
testData_sent = testData_v.split("\n\n")

sum = 0
pathLst_tst = []
for i in range(len(testData_sent)):
    val,path = viterbi(testData_sent[i])
    pathLst_tst.append(path)
    sum = sum + val
#sum






c = 0
with open("viterbi.out", "w") as viterbiFile:
    
    for i in range(len(testData_sent)):
        sent = testData_sent[i].split("\n")
        for j in range(len(sent)):
            s = sent[j].split()
            if j == len(sent) - 1:
                c+=1
                
                viterbiFile.write(str(c))
                viterbiFile.write("\t")
                viterbiFile.write(s[1])
                viterbiFile.write("\t")
                viterbiFile.write(listOfTags[int(pathLst_tst[i][j])])
                
                viterbiFile.write("\n")
                c = 0
                if i != (len(testData_sent)-1):
                    viterbiFile.write("\n")
                
            else:
                c+=1
                viterbiFile.write(str(c))
                viterbiFile.write("\t")
                viterbiFile.write(s[1])
                viterbiFile.write("\t")
                viterbiFile.write(listOfTags[int(pathLst_tst[i][j])])
    
                viterbiFile.write("\n")
                 

viterbiFile.close()

