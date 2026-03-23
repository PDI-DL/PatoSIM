import os
current = __file__

for i in range(3):
    # print(current)
    dir = os.path.dirname(current)
    current = dir
else:
    dir += "/"

print(dir)