def record_target_counts(target):
    dirs = {}
    for i in target:
        if i not in dirs:
            dirs[i] = 1
        else:
            dirs[i] += 1
    return dirs

if __name__ == "__main__":
    import numpy as np
    a = [1,0,0,0,0,0,0,10]
    dirs = record_target_counts(a)
    print(dirs)