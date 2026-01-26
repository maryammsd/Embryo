

# Cleavage data structure
embryo_id = -1
parent_id = -1
cleavage_timestamp = 0
parent_cleavage_timestamp = 0
level = 0  # 1, 2, 4, 8, 16, ...
species_type = ""  # e.g., "mouse", "monkey"
embryo_info = ""  # Additional info about the embryo


class Cleavage:
    def __init__(self, embryo_id, parent_id, cleavage_timestamp, parent_cleavage_timestamp, level, species_type, embryo_info):
        self.embryo_id = embryo_id
        self.parent_id = parent_id
        self.cleavage_timestamp = cleavage_timestamp
        self.parent_cleavage_timestamp = parent_cleavage_timestamp
        self.level = level
        self.species_type = species_type
        self.embryo_info = embryo_info