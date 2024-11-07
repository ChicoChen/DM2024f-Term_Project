import re
import json
from collections import defaultdict

# Dictionary to store the result
path = "./data/apriori/result2_numericLabelized_0.03.txt"
attribute_pairs = defaultdict(list)

# Read the file
with open(path, "r") as file:
    for line in file:
        # Split the line into the number and the attribute set
        num_part, attributes_part = line.strip().split("\t")
        num = float(num_part)

        # Extract individual attributes and their associated launch price categories
        attributes = re.findall(r"([^{,}]+)", attributes_part)
        
        if"Launch price category_" in attributes[0]:
            price = 0
            attrIdx = 1
        else:
            price = 1
            attrIdx = 0

        attribute_pairs[attributes[attrIdx]].append((attributes[price], num))

attribute_pairs = dict(sorted(attribute_pairs.items()))

# Display the results
with open("result2_numericLabelized_0.03.json", "w") as file:
    json.dump(attribute_pairs, file, indent=4)
