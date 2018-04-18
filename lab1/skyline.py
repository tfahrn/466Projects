
# coding: utf-8

# In[189]:


import pandas as pd
from matplotlib import pyplot as plt
import sys

def getReceiptMbs(filename):
    df = pd.read_csv(filename,names = ['recpt_id','quantity','item'])
    
    # mbs: market baskets; maps the receipt number to a set of all the items purchased
    mbs = {}
    for row in df.values:
        item_id = row[2]
        r_id = row[0]

        if(r_id not in mbs):
            mbs[r_id] = set()

        mbs[r_id].add(item_id)
    
    return mbs


# In[190]:


def getItemSets(filename):
    df = pd.read_csv(filename,names = ['recpt_id','quantity','item'])
    
    return set(df['item'])


# In[191]:


# Returns support of itemset
# Checks how many marketbaskets contain the itemset
def getSupport(itemset,mbs):
    count = 0
    for mb in mbs:
        if (itemset.issubset(mb)):
            count+=1
    return count/len(mbs)


# In[192]:


"""
mbs: marketbaskets; map of receipt number to market basket
itemset: set of all items
minSup: minimum support number

return:
"""
def apriori(mbs, itemset, minSup, maxSup):
    F = [] # list of F1, F2, ..., Fn
    F1 = [] # list of all item sets of length 1 where the support of the item set > minSup and < maxSup
    FS1 = [] # list of all item sets of length 1 where support of item set > maxSup
    
    for item in itemset:            
        itemSup = getSupport(set([item]), mbs)
        if(itemSup >= minSup):
            if(itemSup <= maxSup):
                F1.append(set([item]))
            else:
                FS1.append(set([item]))

    F.append(F1)
    
    k = 1 #index to iterate F, eg. F[0] == F1
    while(len(F[k-1]) > 0):
        Ck = candidateGen(F[k-1], k-1) # candidate frequent itemsets of length k+1
        Fk = []
        
        for candidate in Ck:
            count = 0
            for mb in mbs:
                if(candidate.issubset(mb)):
                    count += 1

            if(count/len(mbs) >= minSup):
                Fk.append(candidate)
        
        F.append(Fk)   
        k += 1
                    
    return (F, FS1)


# In[193]:


# Passing in arrray of itemsets of length k
# the size/length of the item sets K
# return: set of candidate frequent item sets of length k+1
def candidateGen(Fk, k):
    candidates = set()
    finalCandidates = set()
    
    #generate candidates of length k+1
    for itemset1 in Fk:
        for itemset2 in Fk:
            # check len(set) == k?
            union = itemset1.union(itemset2)
            if( (itemset1 is not itemset2) and (len(union) == len(itemset1) + 1) ):
                candidates.add(frozenset(union))
    
    #prune candidates
    for cand in candidates:
        isValid = True
        for item in cand:
            prunedCand = set([c for c in cand if c != item])
            if (prunedCand not in Fk):
                isValid = False
                continue;
        if (isValid):
            finalCandidates.add(cand)
            
    return finalCandidates 


# In[194]:


def maximal(itemsets):
    all_itemsets = []
    maximal = []
    
    for itemset_list in itemsets:
        for itemset in itemset_list:
            all_itemsets.append(set(itemset))
    
    for itemset1 in all_itemsets:
        isMaximal = True
        for itemset2 in all_itemsets:
            if itemset1 is not itemset2 and itemset1.issubset(itemset2):
                isMaximal = False
        if isMaximal:
            maximal.append(itemset1)
    
    return maximal


# In[195]:


def genRules(mbs, F, minConf):
    H1 = []
    
    for itemset in F:
        if len(itemset) < 2:
            continue;
        
        for item in itemset:
            # conf = getSupport(itemset, mbs.values())/ getSupport(itemset - set([item]), mbs.values())
            conf = getSupport(itemset, mbs)/ getSupport(itemset - set([item]), mbs)
            if conf >= minConf:
                H1.append([itemset - set([item]), item])
        
    return H1   


# In[196]:


def get_goods(filename):
    goods = pd.read_csv(filename)
    goods = goods[['Flavor','Food']]
    good_labels = []
    for row in goods.values:
        foodItem = row[0].replace("'","") + " " + row[1].replace("'","")
        good_labels.append(foodItem)
    return good_labels


# In[197]:


def report_rules(maxRules,labels, shift):
    translated = []
    for rule in maxRules:
        left = list(rule[0])
        new_left = []
        for item in left:
            new_left.append(labels[item+shift])
        new_right = labels[rule[1]]
        translated.append([new_left,new_right])
    return translated


# In[198]:


def report_itemsets(maxItemsets,labels, shift):
    translated = []
    for itemset in maxItemsets:
        new_itemset = []
        for item in list(itemset):
            new_itemset.append(labels[item+shift])
        translated.append(new_itemset)
    return translated
            


# In[199]:


def bakery_main(filename, goodsfile, minsup, minconf):
    mbs = getReceiptMbs(filename)
    itemsets = getItemSets(filename)
    goods =  get_goods(goodsfile)
    
    SUPPORT = minsup 
    MAX_SUPPORT = 1.0
    CONFIDENCE = minconf 
    (frequent_itemsets, special_itemsets) = apriori(mbs.values(),itemsets, SUPPORT, MAX_SUPPORT)
    maximal_itemsets = maximal(frequent_itemsets)
    labeled_itemsets = report_itemsets(maximal_itemsets,goods, 0)
    #Optimal confidence and support will make 10-50 maximal rules
    maximal_rules = genRules(mbs.values(), maximal_itemsets, CONFIDENCE)
    labeled_rules = report_rules(maximal_rules,goods, 0)
    
    #Report itemsets and supports
    for i in range(len(maximal_itemsets)):
        support = getSupport(maximal_itemsets[i], mbs.values())
        print(labeled_itemsets[i], " has support: ", round(support,3))
        
    #Report rules, support of rules, and confidence of the rules
    rule_supports = []
    rule_confidences = []
    for i in range(len(maximal_rules)):
        itemset = maximal_rules[i][0].union([maximal_rules[i][1]])
        support = getSupport(itemset, mbs.values())
        confidence = getSupport(itemset, mbs.values()) / getSupport((maximal_rules[i][0]),
                                                                    mbs.values())
        rule_supports.append(support)
        rule_confidences.append(confidence)

        print("rule: ", labeled_rules[i][0], "--> ",
              labeled_rules[i][1],
              "\nSupport:",
              round(support,3),
              "Confidence:",
              round(confidence,3))
    
    """
    plt.scatter(rule_confidences, rule_supports)
    plt.xlabel('conf')
    plt.ylabel('sup')
    
    plt.show()
    """


# In[200]:


def bingo_main(bingofile, authorfile, minsup, minconf):
    df = pd.read_csv(authorfile, sep="|",names= ['id','name'])
    f = open(bingofile, "r")
    baskets = []
    itemsets = set(df['id'])
    authors = df['name']
    for line in f:
        tokens = line.split(",")
        basket = tokens[1:]
        basket[len(basket)-1] =  basket[len(basket)-1].replace("\n","")
        basket = [int(item.strip()) for item in basket]
        baskets.append(basket)
    
    SUPPORT = minsup 
    MAX_SUPPORT = 1.0
    CONFIDENCE = minconf 
    (frequent_itemsets, special_itemsets) = apriori(baskets,itemsets, SUPPORT, MAX_SUPPORT)
    maximal_itemsets = maximal(frequent_itemsets)
    labeled_itemsets = report_itemsets(maximal_itemsets,authors, -1)
    maximal_rules = genRules(baskets, maximal_itemsets, CONFIDENCE)
    labeled_rules = report_rules(maximal_rules,authors, -1)
    
    #Report itemsets and supports
    for i in range(len(maximal_itemsets)):
        support = getSupport(maximal_itemsets[i], baskets)
        print(labeled_itemsets[i], " has support: ", round(support,3))

    #Report rules, support of rules, and confidence of the rules
    rule_supports = []
    rule_confidences = []
    for i in range(len(maximal_rules)):
        itemset = maximal_rules[i][0].union([maximal_rules[i][1]])
        support = getSupport(itemset, baskets)
        confidence = getSupport(itemset, baskets) / getSupport((maximal_rules[i][0]),
                                                                    baskets)
        rule_supports.append(support)
        rule_confidences.append(confidence)
        
        print("rule: ", labeled_rules[i][0], "--> ",
              labeled_rules[i][1],
              "\nSupport:",
              round(support,3),
              "Confidence:",
              round(confidence,3))
    """
    plt.scatter(rule_confidences, rule_supports)
    plt.xlabel('conf2')
    plt.ylabel('sup2')
    
    plt.show()
    """
    


# In[201]:


def getGeneMbs(filename):
    df = pd.read_csv(filename,names = ['expgene','tf_id','occurrences'])
    
    # mbs: market baskets; maps the receipt number to a set of all the items purchased
    mbs = {}
    for row in df.values[1:]:
        item_id = int(row[1])
        r_id = int(row[0])

        if(r_id not in mbs):
            mbs[r_id] = set()

        mbs[r_id].add(item_id)
    
    return mbs


# In[202]:


# needs cleanup
def getGeneItemSets(filename):
    df = pd.read_csv(filename,sep=",",names= ['tf_id','transfac'])
    asStrings = set(df['tf_id'][1:])
    return set([int(s) for s in asStrings])


# In[203]:


def getGeneFactors(filename):
    df = pd.read_csv(filename,sep=",",names= ['tf_id','transfac'])
    factors = []
    
    for row in df['transfac'][1:]:
        factors.append(row)
    

    print(factors)
    return factors


# In[204]:


"""
gene: label of market basket (like receipt; 47 in total)
with each gene: collection of transcription factors (like items; 412 in total)
"""
def gene_main(basketfile, factorfile, minsup):

    itemsets = getGeneItemSets(factorfile)
    factors = getGeneFactors(factorfile)
    # mbs: market baskets; maps the gene number to a set of all the transcription factors in it
    mbs = getGeneMbs(basketfile)

    SUPPORT = minsup 
    MAX_SUPPORT = 0.85
    (frequent_itemsets, special_itemsets) = apriori(mbs.values(),itemsets, SUPPORT, MAX_SUPPORT)
    maximal_itemsets = maximal(frequent_itemsets)
    labeled_itemsets = report_itemsets(maximal_itemsets,factors,-1)

    """
    labeled_special = report_itemsets(special_itemsets, factors,-1)
    for i in range(len(special_itemsets)):
        support = getSupport(special_itemsets[i], mbs.values())
        print(labeled_special[i], " has support: ", round(support,3))
    """

    #Report itemsets and supports
    for i in range(len(maximal_itemsets)):
        support = getSupport(maximal_itemsets[i], mbs.values())
        print(labeled_itemsets[i], " has support: ", round(support,3))


# In[205]:


def main():
    filename = sys.argv[1]

    bakery_filenames = ['1000i.csv', '5000i.csv', '20000i.csv', '75000i.csv']

    if(filename in bakery_filenames):
        minsup = float(sys.argv[2])
        minconf = float(sys.argv[3])
        goodsfile = sys.argv[4]
        bakery_main(filename, goodsfile, minsup, minconf)
    elif(filename == 'bingoBaskets.csv'):
        minsup = float(sys.argv[2])
        minconf = float(sys.argv[3])
        authorfile = sys.argv[4]
        bingo_main(filename, authorfile, minsup, minconf)
    elif(filename == 'factor_baskets_full.csv'):
        minsup = float(sys.argv[2])
        factorfile = sys.argv[3]
        gene_main(filename, factorfile, minsup)

    else:
        print("Improper input parameters. First filename must be one of {1000i.csv, 5000i.csv, 20000i.csv, 75000i.csv, bingoBaskets.csv, factor_baskets_full.csv}")
        quit()

# In[206]:


main()

