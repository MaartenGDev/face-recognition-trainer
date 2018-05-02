# Face Training program

## Requirements
- python 3.6
- cv2
- tkinter

## Setup
1. Install all dependencies in `label.py` (`pip3 install Pillow numpy opencv-contrib-python opencv-python`)
2. place the source images(with the faces you want to train) in the `images-to-process` folder
3. Run `python3 label.py`
4. Label the faces


## Usage
1. When the program is started the first image of the `image-to-process` folder to analysed to get the faces in the image.
2. Label the faces by entering the names in the textbox under the faces in the `Detected Faces` panel.
3. Click on next to save the labeled faces to `training-data` and load the next image.
4. The program closes if there are no images left in the `images-to-process` folder.

## Preview


## Licence
MIT