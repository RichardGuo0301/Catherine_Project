from pathlib import Path
from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils_func import make_dir

path = 'models/door_model4.pt'
door_model = YOLO(path)
door_model.conf = 0.45
window_model_path = 'models/window_model.pt'
window_model = YOLO(window_model_path)
window_model.conf = 0.75  # Set your own confidence threshold
people_model = YOLO('models/yolov8n.pt')

# ====== Camera setup for Raspberry Pi ======
piCam = Picamera2()
piCam.preview_configuration.main.size = (1920, 1080)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.resolution = (1920, 1080)
piCam.configure("preview")
piCam.start()

def detect_door(img_path, frame):
    results = door_model(img_path)
    door = []
    annotator = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            # print(b)
            conf = r.boxes.conf.tolist()
            door.append(b)
            c = box.cls
            annotator.box_label(b, door_model.names[int(c)], color=(255, 128, 128))
    frame = annotator.result()
    return door, frame


def detect_people(img_path, frame):
    imgs = [img_path]  # batch of images
    people = []
    # Inference
    results = people_model(imgs, classes=0)
    annotator = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            # print(b)
            conf = r.boxes.conf.tolist()
            people.append(b)
            c = box.cls
            annotator.box_label(b, people_model.names[int(c)], color=(128, 128, 255))
    frame = annotator.result()
    return people, frame

def detect_windows(img_path, frame):
    results = window_model(img_path)
    windows = []
    annotator = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            conf = r.boxes.conf.tolist()
            windows.append(b)
            c = box.cls
            annotator.box_label(b, window_model.names[int(c)], color=(128, 255, 128))
    frame = annotator.result()
    return windows, frame

def write_next2door(image):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method

    image = cv2.putText(image, 'Person -', (5, 50), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Next to door', (5, 80), font, fontScale, color, thickness, cv2.LINE_AA)
    return image


def next2door(person_box, door_box, frame):
    epsilon = 5
    if door_box[0] - epsilon <= person_box[0] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[
                1] <= door_box[3] + epsilon:
        return write_next2door(frame), True

    if door_box[0] - epsilon <= person_box[2] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[1] <= door_box[3] + epsilon:
        return write_next2door(frame)

    if door_box[0] - epsilon <= person_box[0] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[1] <= door_box[3] + epsilon:
        return write_next2door(frame), True

    if door_box[0] - epsilon <= person_box[2] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[3] <= door_box[1] + epsilon:
        return write_next2door(frame), True
    return frame, False

def write_next2window(image):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    image = cv2.putText(image, 'Person -', (5, 130), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Next to window', (5, 160), font, fontScale, color, thickness, cv2.LINE_AA)
    return image

def next2window(person_box, window_box, frame):
    epsilon = 5
    # Check if any corner of the person_box is within epsilon range of the window_box
    is_near_window = (
        (window_box[0] - epsilon <= person_box[0] <= window_box[2] + epsilon and
         window_box[1] - epsilon <= person_box[1] <= window_box[3] + epsilon) or
        (window_box[0] - epsilon <= person_box[2] <= window_box[2] + epsilon and
         window_box[1] - epsilon <= person_box[1] <= window_box[3] + epsilon) or
        (window_box[0] - epsilon <= person_box[0] <= window_box[2] + epsilon and
         window_box[1] - epsilon <= person_box[3] <= window_box[3] + epsilon) or
        (window_box[0] - epsilon <= person_box[2] <= window_box[2] + epsilon and
         window_box[1] - epsilon <= person_box[3] <= window_box[3] + epsilon)
    )

    if is_near_window:
        frame = write_next2window(frame)
    return frame, is_near_window


def process_image(filename, frame, save_image=False):
    # If a filename is provided, read the image from the file
    if filename is not None:
        frame = cv2.imread(filename)

    # Detect doors
    doors, frame = detect_door(frame, frame)

    # Detect windows
    windows, frame = detect_windows(frame, frame)

    # Detect people
    people_list, frame = detect_people(frame, frame)

    is_near_door = False
    is_near_window = False

    # Check if there are any doors detected and if there are people in the frame
    if doors and people_list:
        for person in people_list:
            frame, person_near_door = next2door(person_box=person, door_box=doors[0], frame=frame)
            if person_near_door:
                is_near_door = True

    # Check if there are any windows detected and if there are people in the frame
    if windows and people_list:
        for person in people_list:
            frame, person_near_window = next2window(person_box=person, window_box=windows[0], frame=frame)
            if person_near_window:
                is_near_window = True

    # Save the processed image if required
    if save_image and filename is not None:
        make_dir('result')
        cv2.imwrite(f'result/{Path(filename).stem}.jpg', frame)

    # Return the processed frame along with proximity flags
    return frame, len(people_list), len(doors), len(windows), is_near_door, is_near_window




if __name__ == '__main__':
    path = 'sample_images/door6.jpg'
    img = cv2.imread(path)
    process_image(filename=path, frame=img, save_image=True)


