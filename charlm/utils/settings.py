

def write_settings(file, keys):
    f = open(file, 'wt')
    for k, v in keys.items():
        f.write("%s\t%s\n" % (k, v))
    f.close()


def print_settings(keys):
    for k, v in keys.items():
        print("\t%s: %s" % (k, v))
