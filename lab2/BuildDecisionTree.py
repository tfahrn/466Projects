from anytree import Node, RenderTree

# returns tuple: (most frequent class in D, there is only one class in D)
def get_most_freq_class(D):

    class_to_freq = {}

    for data in D:
        #TODO: find class
        class = data[0]

        if class in class_to_freq:
            class_to_freq[class] = class_to_freq[class] + 1
        else:
            class_to_freq[class] = 1

    total = 0
    max_freq = 0
    max_class = None

    for k, v in class_to_freq.items():

        if v > max_freq:
            max_freq = v
            max_class = k

        total += v

    return (max_class, total == max_freq)

# uses information gain
# returns attr with largest gain else None if all gain < threshold
def select_split_attr_reg(D, A, threshold):
    
    #TODO: build get_entropy()
    entropy = get_entropy(D)
    max_gain = threshold
    best_attr = None

    for attr in A:
        #TODO: build get_entropy()
        entropy_of_split = get_entropy(D, attr)
        gain = entropy - entropy_of_split
        
        if gain > max_gain:
            max_gain = gain
            best_attr = attr

    return best_attr


# uses information gain ratio
def select_split_attr_ratio(D, A, threshold):

    #TODO: build get_entropy()
    entropy = get_entropy(D)
    max_gainRatio = threshold
    best_attr = None

    for attr in A:
        #TODO: build get_entropy()
        entropy_of_split = get_entropy(D, attr)
        gain = entropy - entropy_of_split
        gainRatio = gain / entropy_of_split
        
        if gainRatio > max_gainRatio:
            max_gainRatio = gainRatio
            best_attr = attr

    return best_attr

def build_decision_tree(dataset, attributes, tree, threshold):
    
    most_freq_class, is_only_class = get_most_freq_class(dataset)

    if is_only_class || len(attributes) == 0:
        leaf = Node(most_freq_class)
        tree = leaf
        return tree
    else:
        split_attr = select_split_attr_reg(dataset, attributes, threshold)

        if split_attr is None:
            leaf = Node(most_freq_class)
            tree = leaf
            return tree
        else:
            parent = Node(split_attr)
            
            attr_val_to_data = {}
            for data in dataset:
                # TODO: correct indexing
                attr_val = data[split_attr]

                if attr_val not in attr_val_to_data:
                    attr_val_to_data[attr_val] = []
                attr_val_to_data[attr_val].append(data)

            for k, v in attr_val_to_data:
                child = build_decision_tree(v, attributes.remove(k), null)
                child.parent = parent

            return parent
