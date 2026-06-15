import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/odeleon1/Documents/Projects/Pathfinder/install/depth_anything_node'
