import os
import embryo
import cleavage
import math
import matplotlib.pyplot as plt



def process_csv_line(line, directory_name):    # Dummy implementation for processing a CSV line
    # In a real scenario, parse the line and extract embryo data
    fields = line.strip().split(';')
    if len(fields) < 8:
        return None
    embryo_id = fields[0]
    embryo_selection = fields[1]
    if embryo_selection == 'NA':
        return None
    embryo_timestamp = fields[2]
    embryo_x = fields[3]
    embryo_y = fields[4]
    embryo_z = fields[5]
    embryo_parent_id = fields[6]
    return embryo.Embryo(embryo_id, embryo_timestamp, embryo_x, embryo_y, embryo_z, embryo_parent_id, directory_name)

def read_file(file_path,directory_name=""):
    embryos_data = {}
    parent_child = {}
    timestamp_data = {}
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return
    if file_path.endswith('.csv'):
        # Read CSV file line by line
        with open(file_path, 'r') as file:
            for line in file:
                #print(line.strip())
                # Process each line and extract embryo data
                embryo = process_csv_line(line,directory_name)
                if embryo:
                    embryos_data[embryo.id] = embryo
                    if embryo.parent_id not in parent_child:
                        parent_child[embryo.parent_id] = []
                    if embryo.timestamp not in timestamp_data:
                        timestamp_data[embryo.timestamp] = []
                    parent_child[embryo.parent_id].append(embryo.id)
                    timestamp_data[embryo.timestamp].append(embryo.id)
    elif file_path.endswith('.emb'):
        print("Reading .emb files is not implemented yet.")
    return embryos_data, parent_child, timestamp_data

def get_first_parent(child_id,embryos_data,parent_child):
    current_id = child_id
    while current_id in embryos_data:
        parent_id = embryos_data[current_id].parent_id
        if len(parent_child.get(parent_id, [])) > 1:
            parent_of_parent_id = embryos_data[parent_id].parent_id if parent_id in embryos_data else "N/A"
            print(f"  Found first parent with multiple children: {parent_id} with parent ID: {parent_of_parent_id} and its children: {parent_child[parent_id]}")
            return parent_id
        current_id = parent_id
    return -1

def get_first_grandchild(parent_id,embryos_data,parent_child):
    if parent_id not in parent_child:
        print("  Parent ID not found in parent_child")
        return -1, -1
    current_child = parent_id
    while current_child in parent_child:
        print(f"  Checking children of parent ID: {current_child}")
        print(f"  Number of Children: {len(parent_child[current_child])}")
        if len(parent_child[current_child]) > 1:
            print(f"  Found first grandchild with multiple children: {parent_child[current_child]}")
            grandchild_id = parent_child[current_child][0]
            return current_child,grandchild_id
        elif len(parent_child[current_child])  ==  1:
            current_child = parent_child[current_child][0]
        elif len(parent_child[current_child]) == 0:
            print("  No children found for parent ID :", current_child)
            break
    return -1, -1

def get_cleavage_info(embryos_data, parent_child, timestamp_data):
    cleavage_data = {}
    for parent_id in parent_child:
        if len(parent_child[parent_id]) >  1 :
            
            # what is the parent of this parent with more than one child?
            if parent_id not in embryos_data:
                print("  Parent ID not found in embryos_data")
                continue
            level = math.floor(math.log2(len(timestamp_data[embryos_data[parent_id].timestamp])))
            level = int(math.pow(2,level+1))
            print("--------------------------------------------------")
            print(f"Processing Parent ID: {parent_id} at Level: {level}")
            print(f"  Children: {parent_child[parent_id]}")
            print(f"  Parent Timestamp: {embryos_data[parent_id].timestamp}")
            print(f"  Level: {level}")
            grandparent_id = get_first_parent(parent_id,embryos_data,parent_child)
            if grandparent_id == -1:
                print("  Grandparent ID not found")
                continue
            if int(grandparent_id) > 0 and grandparent_id not in embryos_data:
                print(f"  Grandparent ID not found in embryos_data {grandparent_id} ")
                continue
            initial_grandparent_cleavage = 0
            if int(grandparent_id) != 0:
                for child_id_grand in parent_child[grandparent_id]:
                    if float(embryos_data[child_id_grand].timestamp) < initial_grandparent_cleavage or initial_grandparent_cleavage == 0:
                        initial_grandparent_cleavage = float(embryos_data[child_id_grand].timestamp)
                #nitial_grandparent_cleavage = float(embryos_data[grandparent_id].timestamp)

            # find the minimum cleavage time among all children of this parent
            initial_parent_cleavage = float(embryos_data[parent_child[parent_id][0]].timestamp)
            for child_id in parent_child[parent_id]:
                if float(embryos_data[child_id].timestamp) < initial_parent_cleavage:
                    initial_parent_cleavage = float(embryos_data[child_id].timestamp)

            for child_id in parent_child[parent_id]:
                print(f"  Processing Child ID: {child_id}")
                if child_id not in embryos_data:
                    print(f"  Child ID not found in embryos_data {child_id} ")
                    continue
                
                immediate_parent_id_grand_son, grandson_id = get_first_grandchild(child_id,embryos_data,parent_child)
                if grandson_id == -1 or immediate_parent_id_grand_son == -1:
                    print("  Grandson ID not found")
                    continue
                print(f"  Found grandson ID: {grandson_id}")
                print (f" grandparent_id: {grandparent_id}, parent_id: {parent_id}, child_id: {child_id} , grand_son: {grandson_id} level: {level}")

                # from its birth to its own first child's birth
                child_longetivity = float(embryos_data[grandson_id].timestamp) - float(embryos_data[child_id].timestamp)

                # from its birth to its parent's first child's birth
                parent_longetivity = float(initial_parent_cleavage) - float(initial_grandparent_cleavage)


                print(f" longetivity (parent): {child_longetivity}")
                print(f" longetivity (child): {parent_longetivity}")

                cleavage_data[child_id] = cleavage.Cleavage(
                    embryo_id=child_id,
                    parent_id=parent_id,
                    cleavage_timestamp=child_longetivity + 1,
                    parent_cleavage_timestamp= parent_longetivity + 1,
                    level=level,
                    species_type=embryos_data[child_id].species_type,
                    embryo_info=embryos_data[child_id].name
                )
    if len(cleavage_data) == 0:
        return None
    return cleavage_data

def depict_regression(cleavage_data):
    # x = embryo cleavage_timestamp and y = parent_cleavage_timestamp
    x_values = []
    y_values = []
    for child_id in cleavage_data:
        print(f"Child ID in depiction: {child_id} with cleavage data {cleavage_data[child_id].cleavage_timestamp}, {cleavage_data[child_id].parent_cleavage_timestamp   }")
        cleavage_info = cleavage_data[child_id]
        if cleavage_info.level <= 4 and cleavage_info.level > 16:
            continue
        x_values.append(cleavage_info.cleavage_timestamp)
        y_values.append(cleavage_info.parent_cleavage_timestamp)
    plt.scatter(x_values, y_values)
    plt.xlabel("Embryo Cleavage Timestamp")
    plt.ylabel("Parent Cleavage Timestamp")
    plt.title("Embryo Cleavage vs Parent Cleavage")

    plt.show()

    #save plot as png
    plt.savefig("cleavage_regression.png")

def print_cleavage_data(cleavage_data):
    for child_id in cleavage_data:
        cleavage_info = cleavage_data[child_id]
        print(f"Cleavage Info for Embryo ID {child_id}:")
        print(f"  Parent ID: {cleavage_info.parent_id}")
        print(f"  Cleavage Timestamp: {cleavage_info.cleavage_timestamp}")
        print(f"  Parent Cleavage Timestamp: {cleavage_info.parent_cleavage_timestamp}")
        print(f"  Level: {cleavage_info.level}")
        print(f"  Species Type: {cleavage_info.species_type}")
        print(f"  Embryo Info: {cleavage_info.embryo_info}")
        print("-----")
        depict_regression()

def main():
    print("Let's run the application!")
    # Example file path; in a real scenario, this could be passed as an argument
    file_path = './tracking/A1/cell-fate.csv'
    embryos_data = {}
    parent_child = {}
    timestamp_data = {}
    embryos_data, parent_child, timestamp_data = read_file(file_path)
    cleavage_data = get_cleavage_info(embryos_data, parent_child, timestamp_data)
    print_cleavage_data(cleavage_data)

            
