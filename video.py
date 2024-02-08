import time
from datetime import datetime
import pytz
import numpy as np
import cv2
from database import Database
from image import process_image
from utils_func import remove_images
from picamera2 import Picamera2

db = Database()

def process_video(video_path=0):
    device = 'Device 1'
    # Create a Picamera2 object
    piCam = Picamera2()

    # Configure the camera
    config = piCam.create_preview_configuration(main={"size": (1280, 720), "format": "XRGB8888"})
    piCam.configure(config)
    piCam.start()

    pool = []
    i = 0
    while True:
        # Capture frame-by-frame
        frame = piCam.capture_array()
        # Convert the frame to a format compatible with cv2 (BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        filename = f'images_{i}.jpg'
        # Save the frame to file using cv2.imwrite()
        cv2.imwrite(filename, frame)

        frame, n_people, n_doors, n_windows, is_near_door, is_near_window = process_image(filename, frame, pool)

        # Set the time zone to Pacific Time
        pacific_time_zone = pytz.timezone('US/Pacific')
        current_time_pacific = datetime.now(pacific_time_zone).strftime('%Y-%m-%d %H:%M:%S')
        print(current_time_pacific)

        url = db.upload_file(firebase_path=f"folder_name/img{i%10}.jpg", local_path=filename)
        data = {'Time': current_time_pacific, 'img_url': url, 'near_door': is_near_door,
                'near_window': is_near_window, 'people_detected': n_people, 'door_detected': n_doors,
                'window_detected': n_windows}
        print(data)
        db.set_events(data=data, device=device, collection='events')
        
        time.sleep(2)

        i += 1

    piCam.stop()
    remove_images()

if __name__ == '__main__':
    process_video()


