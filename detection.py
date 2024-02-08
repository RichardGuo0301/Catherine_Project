import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from picamera2 import Picamera2

# ====== YOLO setup for doors and people ======
door_model_path = 'models/window_model.pt'
door_model = YOLO(door_model_path)
door_model.conf = 0.7

people_model_path = 'models/yolov8n.pt'
people_model = YOLO(people_model_path)

# ====== Camera setup for Raspberry Pi ======
piCam = Picamera2()
piCam.preview_configuration.main.size = (1920, 1080)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.resolution = (1920, 1080)
piCam.configure("preview")
piCam.start()

# ====== Functions from the first script ======
def detect_door(img_path, frame):
    results = door_model(img_path)
    door = []
    annotator = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            conf = box.conf  # 获取单个检测框的置信度

            if conf > 0.65:  # 确保使用单个置信度值进行比较
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


def draw_boxes(object_list, image, name, color=(0, 255, 0), thickness=2,
               font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    for start_point, end_point, confidence in object_list:
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        # org
        org = (start_point[0], start_point[1] - 10)
        # Using cv2.putText() method
        image = cv2.putText(image, f' {name} {confidence}%', org, font,
                            font_scale, color, thickness, cv2.LINE_AA)
    return image


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
    image = cv2.putText(image, 'Next to window', (5, 80), font, fontScale, color, thickness, cv2.LINE_AA)
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

# ====== Processing Loop ======
while True:
    # Read a frame from the video
    frame = piCam.capture_array()

    # Detect doors
    doors, frame = detect_door(frame, frame)  # Assuming the detect_door function can take a frame directly

    # Detect people
    people, frame = detect_people(frame, frame)  # Assuming the detect_people function can take a frame directly

    # Initialize the is_near_door flag to False
    is_near_door = False

    # Check if people are near doors
    if doors and people:
        for person in people:
            frame, person_near_door = next2door(person_box=person, door_box=doors[0], frame=frame)
            if person_near_door:
                is_near_door = True

    # Display the annotated frame
    cv2.imshow("Detection", frame)

    # Print the status of is_near_door
    print("Is near door:", is_near_door)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the display window
cv2.destroyAllWindows()

