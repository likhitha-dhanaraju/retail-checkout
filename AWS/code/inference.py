from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import torch
import json
import cv2
import os
import boto3
from PIL import Image
from json import JSONEncoder

from detectron2.utils.visualizer import VisImage, ColorMode, _create_text_labels
import matplotlib as mpl
import matplotlib.colors as mplc
import colorsys
import numpy as np
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from detectron2.utils.colormap import random_color
from sagemaker_containers.beta.framework import worker

_SMALL_OBJECT_AREA_THRESH = 100

class custom_Visualizer:
	def __init__(self, img_rgb, class_names, scale=1.0, instance_mode=ColorMode.IMAGE):
		"""
		Args:
			img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
				the height and width of the image respectively. C is the number of
				color channels. The image is required to be in RGB format since that
				is a requirement of the Matplotlib library. The image is also expected
				to be in the range [0, 255].
			metadata (MetadataCatalog): image metadata.
		"""
		self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
		self.class_names = sorted(class_names)
		self.output = VisImage(self.img, scale=scale)
		self.cpu_device = torch.device("cpu")

		# too small texts are useless, therefore clamp to 9
		self._default_font_size = max(
			np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
		)
		self._instance_mode = instance_mode

	def draw_instance_predictions(self, predictions):
		"""
		Draw instance-level prediction results on an image.
		Args:
			predictions (Instances): the output of an instance detection/segmentation
				model. Following fields will be used to draw:
				"pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
		Returns:
			output (VisImage): image object with visualizations.
		"""
		boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
		scores = predictions.scores if predictions.has("scores") else None
		classes = predictions.pred_classes if predictions.has("pred_classes") else None
		labels = _create_text_labels(classes, scores, self.class_names)
		keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
		masks = None
		colors = None
		alpha = 0.5

		self.overlay_instances(
			masks=masks,
			boxes=boxes,
			labels=labels,
			keypoints=keypoints,
			assigned_colors=colors,
			alpha=alpha,
		)
		return self.output

	def overlay_instances(
		self,
		*,
		boxes=None,
		labels=None,
		masks=None,
		keypoints=None,
		assigned_colors=None,
		alpha=0.5
	):
		"""
		Args:
			boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
				or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
				or a :class:`RotatedBoxes`,
				or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
				for the N objects in a single image,
			labels (list[str]): the text to be displayed for each instance.
			masks (masks-like object): Supported types are:
				* `structures.masks.PolygonMasks`, `structures.masks.BitMasks`.
				* list[list[ndarray]]: contains the segmentation masks for all objects in one image.
					The first level of the list corresponds to individual instances. The second
					level to all the polygon that compose the instance, and the third level
					to the polygon coordinates. The third level should have the format of
					[x0, y0, x1, y1, ..., xn, yn] (n >= 3).
				* list[ndarray]: each ndarray is a binary mask of shape (H, W).
				* list[dict]: each dict is a COCO-style RLE.
			keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
				where the N is the number of instances and K is the number of keypoints.
				The last dimension corresponds to (x, y, visibility or score).
			assigned_colors (list[matplotlib.colors]): a list of colors, where each color
				corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
				for full list of formats that the colors are accepted in.
		Returns:
			output (VisImage): image object with visualizations.
		"""
		num_instances = None
		if boxes is not None:
			boxes = self._convert_boxes(boxes)
			num_instances = len(boxes)
		if labels is not None:
			assert len(labels) == num_instances
		if assigned_colors is None:
			assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
		if num_instances == 0:
			return self.output

		# Display in largest to smallest order to reduce occlusion.
		areas = None
		if boxes is not None:
			areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
		elif masks is not None:
			areas = np.asarray([x.area() for x in masks])

		if areas is not None:
			sorted_idxs = np.argsort(-areas).tolist()
			# Re-order overlapped instances in descending order.
			boxes = boxes[sorted_idxs] if boxes is not None else None
			labels = [labels[k] for k in sorted_idxs] if labels is not None else None
			assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
			

		for i in range(num_instances):
			color = assigned_colors[i]
			if boxes is not None:
				self.draw_box(boxes[i], edge_color=color)

			if labels is not None:
				# first get a box
				if boxes is not None:
					x0, y0, x1, y1 = boxes[i]
					text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
					horiz_align = "left"
				elif masks is not None:
					x0, y0, x1, y1 = masks[i].bbox()

					# draw text in the center (defined by median) when box is not drawn
					# median is less sensitive to outliers.
					text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
					horiz_align = "center"
				else:
					continue  # drawing the box confidence for keypoints isn't very useful.
				# for small objects, draw text at the side to avoid occlusion
				instance_area = (y1 - y0) * (x1 - x0)
				if (
					instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
					or y1 - y0 < 40 * self.output.scale
				):
					if y1 >= self.output.height - 5:
						text_pos = (x1, y0)
					else:
						text_pos = (x0, y1)

				height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
				lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
				font_size = (
					np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
					* 0.5
					* self._default_font_size
				)
				self.draw_text(
					labels[i],
					text_pos,
					color=lighter_color,
					horizontal_alignment=horiz_align,
					font_size=font_size,
				)

		return self.output

	def _convert_boxes(self, boxes):
		"""
		Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
		"""
		if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
			return boxes.tensor.numpy()
		else:
			return np.asarray(boxes)

	def _change_color_brightness(self, color, brightness_factor):
		"""
		Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
		less or more saturation than the original color.
		Args:
			color: color of the polygon. Refer to `matplotlib.colors` for a full list of
				formats that are accepted.
			brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
				0 will correspond to no change, a factor in [-1.0, 0) range will result in
				a darker color and a factor in (0, 1.0] range will result in a lighter color.
		Returns:
			modified_color (tuple[double]): a tuple containing the RGB values of the
				modified color. Each value in the tuple is in the [0.0, 1.0] range.
		"""
		assert brightness_factor >= -1.0 and brightness_factor <= 1.0
		color = mplc.to_rgb(color)
		polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
		modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
		modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
		modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
		modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
		return modified_color

	def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
		"""
		Args:
			box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
				are the coordinates of the image's top left corner. x1 and y1 are the
				coordinates of the image's bottom right corner.
			alpha (float): blending efficient. Smaller values lead to more transparent masks.
			edge_color: color of the outline of the box. Refer to `matplotlib.colors`
				for full list of formats that are accepted.
			line_style (string): the string to use to create the outline of the boxes.
		Returns:
			output (VisImage): image object with box drawn.
		"""
		x0, y0, x1, y1 = box_coord
		width = x1 - x0
		height = y1 - y0

		linewidth = max(self._default_font_size / 4, 1)

		self.output.ax.add_patch(
			mpl.patches.Rectangle(
				(x0, y0),
				width,
				height,
				fill=False,
				edgecolor=edge_color,
				linewidth=linewidth * self.output.scale,
				alpha=alpha,
				linestyle=line_style,
			)
		)
		return self.output

	def draw_text(
		self,
		text,
		position,
		*,
		font_size=None,
		color="g",
		horizontal_alignment="center",
		rotation=0
	):
		"""
		Args:
			text (str): class label
			position (tuple): a tuple of the x and y coordinates to place text on image.
			font_size (int, optional): font of the text. If not provided, a font size
				proportional to the image width is calculated and used.
			color: color of the text. Refer to `matplotlib.colors` for full list
				of formats that are accepted.
			horizontal_alignment (str): see `matplotlib.text.Text`
			rotation: rotation angle in degrees CCW
		Returns:
			output (VisImage): image object with text drawn.
		"""
		if not font_size:
			font_size = self._default_font_size

		# since the text background is dark, we don't want the text to be dark
		color = np.maximum(list(mplc.to_rgb(color)), 0.2)
		color[np.argmax(color)] = max(0.8, np.max(color))

		x, y = position
		self.output.ax.text(
			x,
			y,
			text,
			size=font_size * self.output.scale,
			family="sans-serif",
			bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
			verticalalignment="top",
			horizontalalignment=horizontal_alignment,
			color=color,
			zorder=10,
			rotation=rotation,
		)
		return self.output

def model_fn(model_path):
	device = torch.device('cpu')

	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 200
	cfg.MODEL.DEVICE = 'cpu'

	cfg.MODEL.WEIGHTS = "model_final.pth"
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # The trained model and checkpoints will be saved in the folder /content/output

	predictor = DefaultPredictor(cfg)

	return predictor

def input_fn(request_body, content_type='application/json'):
	if content_type == 'application/json':
		input_data = json.loads(request_body)
		s3 = boto3.resource('s3')
		bucket = s3.Bucket('retail-checkout-data')
		img_object = bucket.Object(input_data['url'])
		response = img_object.get()
		file_stream = response['Body']
		image_data = np.array(Image.open(file_stream))
		
		product_names = input_data['products_list']
		products_list = product_names.strip().split(',')
		
		data = {'image_data':image_data,
				'products_list':products_list,
				'image_name':input_data['url'].strip().split('/')[-1]}
		
		return data
	raise Exception('Requested unsupported ContentType in content_type {content_type}')


def predict_fn(input_data,model):
	outputs = model(input_data['image_data'])

	v = custom_Visualizer(input_data['image_data'][:, :, ::-1],
				   class_names = sorted(input_data['products_list']), 
				   scale=1, 
	)
	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	image = v.get_image()[:,:,::-1]
	s3_client = boto3.client('s3')

	cv2.imwrite(input_data['image_name'],image)
	s3_client.upload_file(input_data['image_name'],'retail-checkout-data','results/'+input_data['image_name'])

	return {'output':outputs,
			'classes':input_data['products_list']}

def output_fn(prediction_output, accept='application/json'):

	#test_metadata = MetadataCatalog.get("rpc_test")
	classes = prediction_output['classes']

	def create_list(instances):
		det_items = [classes[i] for i in instances]
		return det_items

	outputs = prediction_output['output']

	items = outputs["instances"].pred_classes.cpu().detach().numpy()
	items = [str(i) for i in list(items)]
	output= {'result':items}

	if accept == 'application/json':
		return json.dumps(output)
	raise Exception(f'Requested unsupported ContentType in Accept:{accept}')