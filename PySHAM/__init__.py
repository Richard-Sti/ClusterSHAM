
import pickle


def dump_pickle(fname, obj):
    with open(fname, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(fname):
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    return data
