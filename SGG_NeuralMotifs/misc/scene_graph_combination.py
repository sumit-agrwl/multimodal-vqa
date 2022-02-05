import os
import json

# base scene graph
filename = "trainsubset_noattribute_sceneGraphs.json"
base_sceneGraph = json.load(open(filename,'r'))

# part 1 scene graph
filename = "training_fullset_part1.json"
part1_sceneGraph = json.load(open(filename,'r'))

# part 2 scene graph
filename = "training_fullset_part2.json"
part2_sceneGraph = json.load(open(filename,'r'))

save_sceneGraph_dict = {}

for key, value in base_sceneGraph.items():
    save_sceneGraph_dict[key] = value
    
for key, value in part1_sceneGraph.items():
    save_sceneGraph_dict[key] = value
    
for key, value in part2_sceneGraph.items():
    save_sceneGraph_dict[key] = value

import pdb; pdb.set_trace()
print("Done.")