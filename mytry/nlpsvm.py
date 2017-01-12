"""
Parses and represents C4.5 MI data sets
"""
import os
import re
import sys
import traceback
from collections import MutableSequence, defaultdict, Sequence
from itertools import chain, starmap

NAMES_EXT = '.names'
DATA_EXT = '.data'


_A_RE = '\\s*n\\s*|\\s*v\\s*'

class Feature(object):
    """
    Information for a feature
    of NLP data
    """

    class Type:
        """
        Type of feature
        """        
        A = 'A'
        
    def __init__(self, name, ftype, values=None):
        self.name = name
        self.type = ftype
        if values is None:
            self.values = None        
        else:
            raise Exception('Values given for % feature' % self.type)
        self.tup = (self.name, self.type, self.values)

    def __cmp__(self, other):
        if self.tup > other.tup:
            return 1
        elif self.tup < other.tup:
            return -1
        else:
            return 0

    def __hash__(self):
        return self.tup.__hash__()

    def __repr__(self):
        return '<%s, %s, %s>' % self.tup

    def to_float(self, value):
        if value is None:
            return None
        else:
            return value


Feature.A = Feature("A", Feature.Type.A)






class Schema(Sequence):
    """
    Represents a schema for NLP data
    """

    def __init__(self, features):
        self.features = tuple(features)

    def __cmp__(self, other):
        if self.features > other.features:
            return 1
        elif self.features < other.features:
            return -1
        else:
            return 0

    def __hash__(self):
        return self.features.__hash__()

    def __repr__(self):
        return str(self.features)

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self.features.__iter__()

    def __contains__(self, item):
        return self.features.__contains__(item)

    def __getitem__(self, key):
        return self.features[key]


class ExampleSet(MutableSequence):
    """
    Holds a set of examples
    """

    def __init__(self, schema):
        self.schema = schema
        self.examples = []

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return self.examples.__iter__()

    def __contains__(self, item):
        return self.examples.__contains__(item)

    def __getitem__(self, key):
        return self.examples[key]

    def __setitem__(self, key, example):
        if example.schema != self.schema:
            raise ValueError('Schema mismatch')
        self.examples[key] = example

    def __delitem__(self, key):
        del self.examples[key]

    def insert(self, key, example):
        if example.schema != self.schema:
            raise ValueError('Schema mismatch')
        return self.examples.insert(key, example)

    def append(self, example):
        if example.schema != self.schema:
            raise ValueError('Schema mismatch')
        super(ExampleSet, self).append(example)

    def __repr__(self):
        return '<%s, %s>' % (self.schema, self.examples)

    def to_float(self, normalizer=None):
        return [ex.to_float(normalizer) for ex in self]


class Example(MutableSequence):
    """
    Represents a single example
    from a dataset
    """

    def __init__(self, schema):
        self.schema = schema
        self.features = [None for i in range(len(schema))]
        self.weight = 1.0

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self.features.__iter__()

    def __contains__(self, item):
        return self.features.__contains__(item)

    def __getitem__(self, key):
        return self.features[key]

    def __setitem__(self, key, value):
        self.features[key] = value

    def __delitem__(self, key):
        del self.features[key]

    def insert(self, key, item):
        return self.features.insert(key, item)

    def __repr__(self):
        return '<%s, %s, %s>' % (self.schema, self.features, self.weight)

    def copy_of(self):
        ex = Example(self.schema)
        for i, f in enumerate(self):
            ex[i] = f
        return ex

    def to_float(self, normalizer=None):
        if normalizer is None:
            normalizer = lambda x: x
        return normalizer([feature.to_float(value)
                           for feature, value in zip(self.schema, self)])


class Bag(MutableSequence):
    """
    Represents a Bag
    """

    def __init__(self, bag_id, examples):
        classes = map(lambda x: x[-1], examples)
        if any(classes):
            self.label = True
        else:
            self.label = False
        self.bag_id = bag_id
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return self.examples.__iter__()

    def __contains__(self, item):
        return self.examples.__contains__(item)

    def __getitem__(self, key):
        return self.examples[key]

    def __setitem__(self, key, value):
        self.examples[key] = value

    def __delitem__(self, key):
        del self.examples[key]

    def insert(self, key, item):
        return self.examples.insert(key, item)

    def __repr__(self):
        return '<%s, %s>' % (self.examples, self.label)

    def to_float(self, normalizer=None):
        return [example.to_float(normalizer) for example in self]


def bag_set(exampleset, bag_attr=0):
    """
    Construct bags on the given attribute
    """
    bag_dict = defaultdict(list)
    for example in exampleset:
        bag_dict[example[bag_attr]].append(example)
    return [Bag(bag_id, value) for bag_id, value in bag_dict.items()]


def parse_nlp(file_base, rootdir='.'):
    """
    Returns an ExampleSet from the
    NLP formatted data
    """
    schema_name = file_base + NAMES_EXT
    data_name = file_base + DATA_EXT
    schema_file = find_file(schema_name, rootdir)
    if schema_file is None:
        raise ValueError('Schema file not found')
    data_file = find_file(data_name, rootdir)
    if data_file is None:
        raise ValueError('Data file not found')
    return _parse_c45(schema_file, data_file)


def _parse_nlp(schema_filename, data_filename):
    """Parses NLP given file names"""
    try:
        schema = _parse_schema(schema_filename)
    except Exception as e:
        raise Exception('Error parsing schema: %s' % e)

    try:
        examples = _parse_examples(schema, data_filename)
    except Exception as e:
        raise Exception('Error parsing examples: %s' % e)

    return examples


def _parse_schema(schema_filename):
    features = []    
    with open(schema_filename) as schema_file:
        for line in schema_file:
            feature = _parse_feature(line)
            if feature is not None:
                features.append(feature)    
    return Schema(features)


def _parse_feature(line):
    """
    Parse a feature from the given line;
    删除ID
    """
    line = _trim_line(line)
    if len(line) == 0:
        # Blank line
        return None
    if re.match(_A_RE, line) is not None:
        # A feature
        return Feature.A 
        
    remainder = line[1:]
    values = _parse_values(remainder)
    
def _parse_values(remainder):
    values = list()
    for raw in remainder.split(','):
        raw = raw.strip()
        values.append(raw)
    return values


def _parse_examples(schema, data_filename):
    exset = ExampleSet(schema)
    with open(data_filename) as data_file:
        for line in data_file:
            line = _trim_line(line)
            if len(line) == 0:
                continue
            try:
                ex = _parse_example(schema, line)
                exset.append(ex)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                print >> sys.stderr, 'Warning: skipping line: "%s"' % line
    return exset


def _parse_example(schema, line):
    values = _parse_values(line)
    if len(values) != len(schema):
        raise Exception('Feature-data size mismatch: %s' % line)
    ex = Example(schema)
    for i, value in enumerate(values):
        if value == '?':
            # Unknown value says 'None'
            continue
        stype = schema[i].type
        elif value == '%':
            continue
        stype = schema[i].type
        else:
            raise ValueError('Unknown schema type "%s"' % stype)
    return ex


def _trim_line(line):
    """
    Removes \n
    from the given line
    """
    line = line.strip()
    if len(line) > 0 :
        return line


def find_file(filename, rootdir):
    """
    Finds a file with filename located in
    some subdirectory of the current directory
    """
    for dirpath, _, filenames in os.walk(rootdir):
        if filename in filenames:
            return os.path.join(dirpath, filename)


def save_nlp(example_set, basename, basedir='.'):
    schema_name = os.path.join(basedir, basename + NAMES_EXT)
    data_name = os.path.join(basedir, basename + DATA_EXT)

    print schema_name
    with open(schema_name, 'w+') as schema_file:
        schema_file.write('0,1.\n')
        
    with open(data_name, 'w+') as data_file:
        for example in example_set:
            ex_strs = starmap(_feature_to_str, zip(example.schema, example))
            data_file.write('%s.\n' % ','.join(ex_strs))



