import cv2
import numpy as np
import pyautogui
import time
import logging
import pygame

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
pygame.mixer.init()

# Configuration for different monitor areas, icons, and sounds
monitor_config = [
    {
        'area': (1700, 850, 320, 150),
        'icon_path': 'targeticon.png',
        'sound_path': 'targetalert.mp3'
    },
    {
        'area': (2020, 1650, 400, 140),
        'icon_path': 'swordicon.png',
        'sound_path': 'swordalert.mp3'
    }
]

debug_mode = True  # Toggle debug mode
target_height = 150  # Uniform height for image resizing
window_title = 'Press Q to quit'  # Title of the OpenCV window
threshold = 0.7  # Matching threshold

def capture_and_resize(area):
    screenshot = pyautogui.screenshot(region=area)
    screen_capture = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    resized_capture = cv2.resize(screen_capture, (screen_capture.shape[1], target_height))
    return resized_capture

def icon_visible(reference_icon, screen_capture, config_name):
    reference_icon = cv2.imread(reference_icon, cv2.IMREAD_UNCHANGED)
    res = cv2.matchTemplate(screen_capture, reference_icon, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(screen_capture, pt, (pt[0] + reference_icon.shape[1], pt[1] + reference_icon.shape[0]), (0, 255, 0), 2)
    return len(loc[0]) > 0

def play_alert_sound(sound_path):
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)

try:
    while True:
        combined_images = None
        for config in monitor_config:
            screen_capture = capture_and_resize(config['area'])
            if icon_visible(config['icon_path'], screen_capture, config['icon_path']):
                play_alert_sound(config['sound_path'])
            if combined_images is None:
                combined_images = screen_capture
            else:
                combined_images = np.hstack((combined_images, screen_capture))

        if debug_mode and combined_images is not None:
            cv2.imshow(window_title, combined_images)
            cv2.resizeWindow(window_title, combined_images.shape[1], combined_images.shape[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.1)

except KeyboardInterrupt:
    logging.info("Script interrupted by user.")
finally:
    cv2.destroyAllWindows()
