
def unique(l):
    output = []
    for x in l:
        if x not in output:
            output.append(x)
    return output