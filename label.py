from tkinter import *
from tkinter import filedialog
import uuid
from pathlib import Path

from image_processing import *
from PIL import ImageTk
import os



class App:
    image_source_dir = "test-data/"
    training_data_dir = "training-data/"
    app_root = os.getcwd() + "/"
    output_folder = app_root + "output/"

    images_to_process = []
    image_to_process = None
    image_to_process_index = 0

    detected_faces_photos = []
    shown_faces_paths = []
    shown_faces_label_inputs = []

    current_scaled_source_image = None

    def __init__(self, master):
        master.wm_title('File Parser')
        self.image_preview_frame = LabelFrame(master, text=" Current Image: ")
        self.image_preview_frame.grid(row=0, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        controls_frame = LabelFrame(master, text=" Controls: ")
        controls_frame.grid(row=0, column=9, columnspan=2, rowspan=8, sticky='NS', padx=5, pady=5, ipadx=10, ipady=5)

        Button(controls_frame, text="Next", command=self.label_faces).grid(row=0, padx=10)
        Button(controls_frame, text="Quit", fg="red", command=master.quit).grid(row=1, padx=10)

        # Face preview
        self.face_canvas = None

        # Face labels
        self.face_labels_frame = LabelFrame(master, text=" Detected Faces: ")
        self.face_labels_frame.grid(row=2, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        self.master = master
        self.load_source_folder()

    def load_source_folder(self):
        selected_dir = self.app_root + "images-to-process/"
        files = os.listdir(selected_dir)

        for file in files:
            file = file.lower()
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                self.images_to_process.append(selected_dir + file)

        self.loop_through_images(self.images_to_process)

    def loop_through_images(self, images):
        print(str(self.image_to_process_index + 1) + "/" + str(len(self.images_to_process)))

        self.show_image_using_path(images[self.image_to_process_index])

    def show_next_image(self):
        self.image_to_process_index += 1
        if(self.image_to_process_index > len(self.images_to_process)):
            self.master.quit()

        self.loop_through_images(self.images_to_process)

    def get_scaled_path(self, path):
        image_to_resize = cv2.imread(path)
        scaled_image = resize_to_fit(image_to_resize, 1080, 720)
        scaled_image_path = self.app_root + "output/scaled-image.png"
        cv2.imwrite(scaled_image_path, scaled_image)

        return scaled_image_path

    def show_image_using_path(self, path):
        self.image_to_process = ImageTk.PhotoImage(file=self.get_scaled_path(path))
        self.face_canvas = self.create_image_preview(self.image_preview_frame, self.image_to_process)
        self.face_canvas.grid(row=0, columnspan=7)

        self.show_faces_in_current_image(path)

    def show_faces_in_current_image(self, image_path):
        for child in self.face_labels_frame.winfo_children():
            child.destroy()

        self.face_labels_frame.grid(row=2, columnspan=7, sticky='WE', padx=5, pady=5, ipadx=5, ipady=5)

        self.shown_faces_paths.clear()
        self.detected_faces_photos.clear()
        self.shown_faces_label_inputs.clear()

        raw_image_dimensions = ImageTk.PhotoImage(file=image_path)

        width_scale = self.image_to_process.width() / raw_image_dimensions.width()
        height_scale = self.image_to_process.height() / raw_image_dimensions.height()

        for index, (grey_photo, face_rect) in enumerate(self.detect_faces(cv2.imread(image_path))):
            (x, y, w, h) = face_rect

            self.face_canvas.create_rectangle(x * width_scale, y * height_scale, (x + w) * width_scale, (y + h) * height_scale)

            # To show in the UI
            scaled_image = cv2.resize(grey_photo, (64, 64))
            scaled_image_path = self.output_folder + str(index) + "-grey-image_scaled.png"
            cv2.imwrite(scaled_image_path, scaled_image)

            # To train the model
            grey_photo_path = self.output_folder + str(index) + "-grey-image.png"
            self.shown_faces_paths.append(grey_photo_path)
            cv2.imwrite(grey_photo_path, grey_photo)

            # Detected face preview with label
            self.photo_image = ImageTk.PhotoImage(file=scaled_image_path)
            self.detected_faces_photos.append(self.photo_image)
            self.face_preview = self.create_image_preview(self.face_labels_frame, self.detected_faces_photos[index])
            self.shown_faces_label_inputs.append(Entry(self.face_labels_frame, width=10))

            os.remove(scaled_image_path)

            # Show input to label image
            self.face_preview.grid(row=0, column=index)
            self.shown_faces_label_inputs[index].grid(row=1, column=index)

        os.rename(image_path, self.app_root + "processed-images/" + Path(image_path).name)

    def create_image_preview(self, master, image):
        canvas = Canvas(master, width=image.width(), height=image.height())
        canvas.create_image((0, 0), image=image, anchor='nw')

        return canvas

    def label_faces(self):
        for index, face_path in enumerate(self.shown_faces_paths):
            folder_name = self.shown_faces_label_inputs[index].get()
            storage_folder = self.training_data_dir + folder_name + "/"

            if folder_name != "":
                if not os.path.exists(storage_folder):
                    os.makedirs(storage_folder)

                os.rename(face_path, storage_folder + str(uuid.uuid4()) + ".png")

        self.show_next_image()

    # returns all gray scaled images and rects
    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

        if (len(faces) == 0):
            return []

        face_results = []

        for face in faces:
            (x, y, w, h) = face
            face_results.append((gray[y:y + w, x:x + h], face))

        return face_results


root = Tk()
app = App(root)

root.mainloop()
root.destroy()  # optional; see description below
