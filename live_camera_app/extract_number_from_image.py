import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from models.research.object_detection.utils import visualization_utils as vis_util
from models.research.object_detection.utils import label_map_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_number_and_sorted_digits(boxes, scores, classes, category_index, prob_threshold=0.5):
    # box = [y1, x1, y2, x2]
    # count digits
    n_digits = 0
    for score in scores:
        if score < prob_threshold:
            break
        n_digits += 1
    # sort digits
    digits = [(box, score, _class)
              for box, score, _class in zip(boxes[:n_digits], scores[:n_digits], classes[:n_digits])]
    digits = sorted(digits, key=lambda digit: digit[0][1])
    # get number
    number_str = ''
    for digit in digits:
        number_str += category_index[digit[2]]['name']
    return number_str, digits


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    # PATH_TO_IMAGES_DIR
    parser.add_argument("-images_dir", type=str, dest='path_to_images_dir',
                        default=os.path.join('data', 'test', '1.png'),
                        help="Path to images dir or path to one image. Image formats: .png, .jpg.")
    # OUTPUT_DIR
    parser.add_argument("-output_dir", type=str, dest='output_dir',
                        default=os.path.join('recognized_images'),
                        help="Path to dir where will be saved images with digits' borders.")
    # PATH_TO_CKPT
    parser.add_argument("-path_to_ckpt", type=str, dest='path_to_ckpt',
                        default=os.path.join('exported_model_directory', 'frozen_inference_graph.pb'),
                        help="Path to frozen detection graph. \
                        This is the actual model that is used for the object detection.")
    return parser.parse_args()


def get_image_paths(path_to_images_dir):
    image_formats = ('.png', '.jpg')
    image_paths = []
    if os.path.isdir(path_to_images_dir):
        # directory
        image_paths = [
            os.path.join(path_to_images_dir, name)
            for name in filter(lambda file_name: file_name.endswith(image_formats), os.listdir(path_to_images_dir))
        ]
    elif path_to_images_dir.endswith(image_formats):
        # file
        image_paths.append(path_to_images_dir)
    return image_paths


def load_model(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_layers(detection_graph):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    layers_list = [detection_boxes, detection_scores, detection_classes, num_detections]
    return image_tensor, layers_list


def process_image(image_path, sess, image_tensor, layers_list, category_index, output_dir):
    img_index = int(image_path.split('/')[-1].split('.')[0])
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        layers_list, feed_dict={image_tensor: image_np_expanded})
    number_str, digits = get_number_and_sorted_digits(
        np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), category_index)
    (boxes, scores, classes) = [array for array in zip(*digits)]
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(boxes),
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1)
    img = Image.fromarray(image_np)
    img.save(os.path.join(output_dir, str(img_index) + '_(' + number_str + ')' + '.png'))
    print("On image: '{}' is shown number: {}".format(image_path, number_str))


def main():
    # Set args
    args = parse_args()
    path_to_labels = 'label_map.pbtxt'
    n_classes = 10
    path_to_images_dir = args.path_to_images_dir
    output_dir = args.output_dir
    path_to_ckpt = args.path_to_ckpt

    # Full paths of each images in the directory
    image_paths = get_image_paths(path_to_images_dir)

    # Names of classes
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load model
    detection_graph = load_model(path_to_ckpt)

    # Calculate
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor, layers_list = get_layers(detection_graph)
            for image_path in image_paths:
                process_image(image_path, sess, image_tensor, layers_list, category_index, output_dir)


if __name__ == '__main__':
    main()
