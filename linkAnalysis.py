import numpy as np
import pandas as pd
import os

from itertools import chain, combinations
from collections import defaultdict, namedtuple

from optparse import OptionParser
def formDataSet(dataframe, price_column, numeric_columns):
    bin_info = {}
    for col in numeric_columns:
        dataframe[col], bins = pd.qcut(dataframe[col], q=4, labels=False, retbins=True)
        bin_info[col] = bins  # Store bin edges for the column

    print(bin_info)

    for col in dataframe.columns:
        dataframe[col] = dataframe[col].apply(lambda x: f"{col}_{x}")
    dataframe = dataframe.iloc[:, 1:]
    
    new_rows = []
    
    # Iterate through each row in the original DataFrame
    for idx, row in dataframe.iterrows():
        price = row[price_column]
        
        # Get all columns except the price column
        element_columns = [col for col in dataframe.columns if col != price_column]
        
        # Create a new row for each element-price pair
        for col in element_columns:
            new_rows.append({
                'element': row[col],
                'price': price
            })
    
    # Create new DataFrame from the collected rows
    return pd.DataFrame(new_rows)

    # return dataframe

def inputGenerator(list):
    for line in list:
        record = frozenset(line) # item sets
        yield record

def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = record
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
            
    return itemSet, transactionList

def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    #? join set, but never prune?
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )

def prune(CSet, lastLSet, length):
    prunedSet = set()
    for candidate in CSet:
        pruned = False
        for subset in combinations(candidate, length - 1):
            if frozenset(subset) not in lastLSet:
                pruned = True
                break
        if(not pruned): prunedSet.add(candidate)
    return prunedSet

def runApriori(itemSet, transactionList, minSupport):
    """
    run the apriori algorithm, return.
     - items (tuple, support)
    """
    freqSet = defaultdict(int)
    largeSet = dict()
    candidateCount = defaultdict(int)
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    currentLSet = oneCSet #Large set
    candidateCount[1] = (len(itemSet), len(oneCSet))

    k = 2
    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet

        # generate and prune candidate set
        currentCSet = joinSet(currentLSet, k)
        beforePruning = len(currentCSet)
        currentCSet = prune(currentCSet, currentLSet, k)
        currentLSet = returnItemsWithMinSupport(
            currentCSet, transactionList, minSupport, freqSet
        )

        afterPruning = len(currentLSet)
        candidateCount[k] = (beforePruning, afterPruning)

        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems, candidateCount

def exportResults(items, filepath, filename, factor):
    """output the generated itemsets sorted by support as file"""
    ouputDir = filepath + '/' + filename
    print(f"write to {filepath}")
    if not os.path.exists(filepath) and len(filepath) > 0:
        os.makedirs(filepath)

    with open(ouputDir, 'w+') as f:
        for item, support in sorted(items, key=lambda x: x[1], reverse=True):
            if(len(item) > 1):
                f.write(f"{support * 100 * factor:.1f}\t{{{','.join(item)}}}\n")

if __name__ == "__main__":
    import time
    
    data = pd.read_csv("./data/processed(merged version).csv")
    print(f"data.shape: {data.shape}")

    factor = data.shape[1] - 2
    #"Touch sampling rate" and "Max rated brightness", "Matrix (megapixels)" seems irrelated in current result
    numeric_columns = \
        ["Total Score", "CPU", "GPU", "Memory", "UX", "PPI", \
        "Height (mm)", "Width (mm)", "Thickness (mm)", "Weight (gr)", \
        "Phone age in days"]
    df = formDataSet(data, "Launch price category", numeric_columns)
    print(f"new data.shape: {df.shape}")
    df = df.values.tolist()
    records = inputGenerator(df)

    optparser = OptionParser()
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.01,
        type="float",
    )
    (options, args) = optparser.parse_args()
    minSupport = options.minS / factor
    startTime = time.process_time()

    itemSet, transactionList = getItemSetTransactionList(records)
    items, candidates = runApriori(itemSet, transactionList, minSupport)
    endTime = time.process_time()
    exportResults(items, "./data/apriori", "result2_numericLabelized_0.03.txt", factor)
    print(f"execution time: {endTime - startTime}")

