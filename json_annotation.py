import json
import numpy as np
import os
import sys

import config

from utils import getFilenames


json_content = {
   "_via_settings":{
      "ui":{
         "annotation_editor_height":25,
         "annotation_editor_fontsize":0.8,
         "leftsidebar_width":18,
         "image_grid":{
            "img_height":80,
            "rshape_fill":"none",
            "rshape_fill_opacity":0.3,
            "rshape_stroke":"yellow",
            "rshape_stroke_width":2,
            "show_region_shape":True,
            "show_image_policy":"all"
         },
         "image":{
            "region_label":"__via_region_id__",
            "region_color":"__via_default_region_color__",
            "region_label_font":"10px Sans",
            "on_image_annotation_editor_placement":"NEAR_REGION"
         }
      },
      "core":{
         "buffer_size":18,
         "filepath":{
            
         },
         "default_filepath":""
      },
      "project":{
         "name":"via_project_8May2025_22h50m"
      }
   },
   "_via_img_metadata":{},
   "_via_attributes":{
      "region":{
         
      },
      "file":{
         
      }
   },
   "_via_data_format_version":"2.0.10",
   "_via_image_id_list":[]
}

class WingImageAnnotation:
   def __init__(self, filename, size, rects):
      self.filename = filename
      self.size = size
      self.rects = rects

def createReactReagion(rect):
	return {
		"shape_attributes":{
			"name":"rect",
         "x":rect["x"],
         "y":rect["y"],
         "width":rect["width"],
         "height":rect["height"]
		 },
	   "region_attributes":{}
	  }

def createMetadata(filename, size, rects):
	regions = [createReactReagion(react) for react in rects]

	return {
		f"{filename}{size}": {
			"filename": filename,
			"size": size,
			"regions": regions,
			"file_attributes":{},
		}
	}

def main():
   filenameImages = getFilenames(config.LANDMARKS_PATH + "/images/original", config.DATASET_IN_EXTENSION)
   filenameNpys = getFilenames(config.LANDMARKS_PATH, config.NPY_EXTENSION)

   filenameImages.sort()
   filenameNpys.sort()

   wingImageAnnotations = []
   	
   for i in range(len(filenameImages)):
      filename = filenameImages[i].replace(config.LANDMARKS_PATH + "/images/original/", "")
      size = os.path.getsize(filenameImages[i])
      rects = []

      landmarksNpy = np.load(filenameNpys[i], allow_pickle=True).tolist()
      half = 22
	  
      for landmark in landmarksNpy:
         rects.append({"x":landmark[0] - half, "y":landmark[1] - half, "width":44, "height":44})

      wingImageAnnotations.append(WingImageAnnotation(filename, size, rects))


	
   for wingImageAnnotation in wingImageAnnotations:
      metadata = createMetadata(wingImageAnnotation.filename, wingImageAnnotation.size, wingImageAnnotation.rects)
      id_list = [f"{wingImageAnnotation.filename}{wingImageAnnotation.size}"]
      
      json_content["_via_img_metadata"].update(metadata)
      json_content["_via_image_id_list"].extend(id_list)



   n = len(sys.argv)
   indent = 4 if n == 2 and "--pretty" == sys.argv[1] else None
      
   json_object = json.dumps(json_content, indent = indent)
	
   with open("./annotation/sample.json", "w") as outfile:
	   outfile.write(json_object)

	
main()