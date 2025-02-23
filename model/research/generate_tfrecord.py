import os
import tensorflow as tf
from object_detection.utils import dataset_util
from lxml import etree
import io

def xml_to_tf_example(xml_path, img_dir, label_map):
    with tf.io.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    print(data)
    image_path = os.path.join(img_dir, data['filename'])
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    classes = []

    for obj in data['object']:
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))

        # Map class name to numeric ID using the label_map
        if obj['name'] in label_map:
            classes.append(label_map[obj['name']])
        else:
            raise ValueError(f"Class name '{obj['name']}' not found in label map.")

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(annotations_dir, img_dir, output_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    # print(os.listdir(annotations_dir))
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue  # Skip non-XML files
        xml_path = os.path.join(annotations_dir, xml_file)
        try:
            tf_example = xml_to_tf_example(xml_path, img_dir, label_map)
            writer.write(tf_example.SerializeToString())
        except Exception as e:
            print(f'Error processing {xml_file}: {e}')
            
    writer.close()

# Define the label map
label_map = {
    'blue helmet': 1,
    'head': 2,
    'red helmet': 3,
    'white helmet': 4,
    'yellow helmet': 5,
    
}

create_tf_record('./data/train', './data/train', '../ssd_mobilenet_v2/train/train.record', label_map)
create_tf_record('./data/valid', './data/valid', '../ssd_mobilenet_v2/train/val.record', label_map)
create_tf_record('./data/test', './data/test', '../ssd_mobilenet_v2/train/test.record', label_map)
