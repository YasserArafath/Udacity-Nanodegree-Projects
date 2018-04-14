
# coding: utf-8

# In[1]:



# In[ ]:

import csv
import codecs
import re
import xml.etree.cElementTree as ET
import cerberus
import schema
from audit import update_name, is_street_name

OSM_PATH = "ottawa.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
PHONENUM = re.compile(r'\+1\s\d{3}\s\d{3}\s\d{4}')
POSTCODE = re.compile(r'[A-z]\d[A-z]\s?\d[A-z]\d')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

def update_phone_num(phone_num):
    # Check for valid phone number format
    m = PHONENUM.match(phone_num)
    if m is None:
        # Convert all dashes to spaces
        if "-" in phone_num:
            phone_num = re.sub("-", " ", phone_num)
        # Remove all brackets
        if "(" in phone_num or ")" in phone_num:
            phone_num = re.sub("[()]", "", phone_num)
        # Space out 10 straight numbers
        if re.match(r'\d{10}', phone_num) is not None:
            phone_num = phone_num[:3] + " " + phone_num[3:6] + " " + phone_num[6:]
        # Space out 11 straight numbers
        elif re.match(r'\d{11}', phone_num) is not None:
            phone_num = phone_num[:1] + " " + phone_num[1:4] + " " + phone_num[4:7] + " " + phone_num[7:]
        # Add full country code
        if re.match(r'\d{3}\s\d{3}\s\d{4}', phone_num) is not None:
            phone_num = "+1 " + phone_num
        # Add + in country code
        elif re.match(r'1\s\d{3}\s\d{3}\s\d{4}', phone_num) is not None:
            phone_num = "+" + phone_num
        # Ignore tag if no area code and local number (<10 digits)
        elif sum(c.isdigit() for c in phone_num) < 10:
            return None
    return phone_num


def load_new_tag(element, secondary, default_tag_type):
    new = {}
    new['id'] = element.attrib['id']
    if ":" not in secondary.attrib['k']:
        new['key'] = secondary.attrib['k']
        new['type'] = default_tag_type
    else:
        post_colon = secondary.attrib['k'].index(":") + 1
        new['key'] = secondary.attrib['k'][post_colon:]
        new['type'] = secondary.attrib['k'][:post_colon - 1]

    # Cleaning and loading values of various keys
    if is_street_name(secondary):
        # Why don't i need to use mapping, street_mapping,
        # and num_line_mapping dicts  as params?
        street_name = update_name(secondary.attrib['v'])
        new['value'] = street_name
    
    elif new['key'] == 'phone':
        phone_num = update_phone_num(secondary.attrib['v'])
        if phone_num is not None:
            new['value'] = phone_num
        else:
            return None
    
    elif new['key'] == 'province':
        # Change Ontario to ON
        province = secondary.attrib['v']
        if province == 'Ontario':
            province = 'ON'
        new['value'] = province

    elif new['key'] == 'postcode':
        post_code = secondary.attrib['v'].strip()
        m = POSTCODE.match(post_code)
        if m is not None:
            # Add space in middle if there is none
            if " " not in post_code:
                post_code = post_code[:3] + " " + post_code[3:]
            # Convert to upper case
            new['value'] = post_code.upper()
        else:
            # Keep zip code revealed in postal code audit for document deletion purposes
            if post_code[:5] == "14174":
                new['value'] = post_code
            # Ignore tag if improper postal code format
            else:
                return None

    else:
        new['value'] = secondary.attrib['v']
    
    return new


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements

    if element.tag == 'node':
        for attrib, value in element.attrib.iteritems():
            if attrib in node_attr_fields:
                node_attribs[attrib] = value
        
        # For elements within the top element
        for secondary in element.iter():
            if secondary.tag == 'tag':
                if problem_chars.match(secondary.attrib['k']) is not None:
                    continue
                else:
                    new = load_new_tag(element, secondary, default_tag_type)
                    if new is not None:
                        tags.append(new)
        return {'node': node_attribs, 'node_tags': tags}
    
    elif element.tag == 'way':
        for attrib, value in element.attrib.iteritems():
            if attrib in way_attr_fields:
                way_attribs[attrib] = value

        counter = 0
        for secondary in element.iter():
            if secondary.tag == 'tag':
                if problem_chars.match(secondary.attrib['k']) is not None:
                    continue
                else:
                    new = load_new_tag(element, secondary, default_tag_type)
                    if new is not None:
                        tags.append(new)
            elif secondary.tag == 'nd':
                newnd = {}
                newnd['id'] = element.attrib['id']
                newnd['node_id'] = secondary.attrib['ref']
                newnd['position'] = counter
                counter += 1
                way_nodes.append(newnd)
        
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


#HELPER FUNCTION
def get_element(osm_file, tags=('node', 'way', 'relation')):
    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_strings = (
            "{0}: {1}".format(k, v if isinstance(v, str) else ", ".join(v))
            for k, v in errors.iteritems()
        )
        raise cerberus.ValidationError(
            message_string.format(field, "\n".join(error_strings))
        )


class UnicodeDictWriter(csv.DictWriter, object):
    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })
    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


#MAIN FUNCTION
def process_map(file_in, validate):
    with codecs.open(NODES_PATH, 'w') as nodes_file,          codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file,          codecs.open(WAYS_PATH, 'w') as ways_file,          codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file,          codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    process_map(OSM_PATH, validate=True)

