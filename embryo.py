

# Embryo data structure
id = -1
parent_id = -1
name = ""
x = 0
y = 0
z = 0
timestamp = 0
parent_cleavage_timestamp = 0
level = 0  # 1, 2, 4, 8, 16, ...
species_type = "" # e.g., "mouse", "monkey"

class Embryo:
    def __init__(self, embryo_id, timestamp, x, y, z, parent_id, name,species_type=""):
        self.id = embryo_id
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.parent_id = parent_id
        self.name = name
        self.species_type = species_type