#-------------------------------------------------------
# This program drives the simulated car through various
# lane scenarios, captures camera views, automatically 
#determines ground truth labels (steering/offset), 
# and saves this data.
# --- DATA GENERATION CONFIGURATION NOTES ---
# The current simulation parameters (e.g., random steering frequency/magnitude,
# steering correction strength, and car X-position reset upon loop)
# are tuned to reliably generate ~6400 images
# where both lane lines (yellow and white dashed) remain consistently within
# the camera's field of view.

# If a significantly larger dataset (e.g., tens or hundreds of thousands of images)
# is required, these parameters, particularly those controlling random steering
# and car correction/boundary behavior, may need to be re-evaluated and adjusted
# to maintain visual consistency across a longer simulation run without lines
# disappearing or the car leaving the road.

#-------------------------------------------------------
# import components from simulator.py and other libraries.
import pygame
import numpy as np
import os
from PIL import Image # Only if we want to save images as PNG/JPG

# Import simulator components
from simulator import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    WHITE, BLACK, GRAY, YELLOW,
    LANE_WIDTH, ROAD_WIDTH, LANE_LINE_WIDTH,
    CAR_WIDTH, CAR_HEIGHT, CAR_SPEED, CAR_STEERING_SPEED,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_OFFSET_Y, CAMERA_Y_OFFSET_FROM_CAR_CENTER, #CAMERA_Y_RELATIVE_TO_CAR_FRONT,
    Car, draw_road, draw_lane_lines, get_camera_view
)

# --- Data Generation Constants
DATA_DIR = "data" # To run the script from the root directory
IMAGES_SUBDIR = "images"
LABELS_FILE = "labels.csv"
NUM_SAMPLES = 6400 # 5000 Target number of training samples
#-------------------------------------------------------Automated Driving Logic

def generate_straight_road_data(screen, clock, car, num_samples):
    """
    Define how the car "drives" to generate data. Collect data for
    -Car centered in the lane.
    -Car slightly off-center (left/right).
    """
    print(f"Generating {num_samples} samples for straight road...")

    print(f"Current Working Directory: {os.getcwd()}")

    os.makedirs(os.path.join(DATA_DIR, IMAGES_SUBDIR), exist_ok=True)
    labels_filepath = os.path.join(DATA_DIR, LABELS_FILE)

    with open(labels_filepath, 'w') as f:
        f.write("image_filename,steering_angle\n")

    samples_generated = 0
    while samples_generated < num_samples:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # --- Update Car Position ---
        car.move()

        # --infinite straight road  ---
        # If the car moves off the top of the screen, reset it to the bottom
        # This creates the illusion of an infinite straight road
        if car.y < -CAR_HEIGHT: # Car is completely off the top
            car.y = SCREEN_HEIGHT + CAR_HEIGHT / 2 # Reset to below the screen, or at the bottom
        # --- END infinite BLOCK ---
        # --- RESET X POSITION WITH A SMALL RANDOM OFFSET ---
            # This forces the car back closer to the center upon each loop
            car.x = SCREEN_WIDTH / 2 + np.random.uniform(-20, 20) # Reset to center +/- 20 pixels
            # --- END of RESET X POSITION ---

        # --- Simulate Car Deviations and Keep it on Road ---
        # Introduce slight random steering for diversity
        if np.random.rand() < 0.03: # Changed from 0.2 to 0.03 (e.g., 3% chance per frame)
            car.steer(np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.2)) # 1.0)) # Smaller random steer for now

        # Define safe driving bounds for the car's X position
        # These should be slightly inside the drawn road to avoid capturing black edges
        safe_left_bound = SCREEN_WIDTH / 2 - ROAD_WIDTH / 2 + CAR_WIDTH / 2 # +10 px buffer: too much
        safe_right_bound = SCREEN_WIDTH / 2 + ROAD_WIDTH / 2 - CAR_WIDTH / 2 # -10 px buffer: too much

        # If car goes outside safe bounds, gently steer it back towards center
        if car.x < safe_left_bound:
            car.steer(-3) # Changed -2 to -3 # Steer right
        elif car.x > safe_right_bound:
            car.steer(3) # Steer left: changed 2 to 3
        
        # Also ensure car angle doesn't drift too far from straight ahead
        # If car's angle deviates significantly, correct it back towards straight (90 degrees)
        if car.angle > 95: # <--- Changed from 100 to 95 (reacts sooner)
            car.steer(-2) # <--- Changed from -1 to -2 (stronger correction)
        elif car.angle < 85: # <--- Changed from 80 to 85 (reacts sooner)
            car.steer(2) # <--- Changed from 1 to 2 (stronger correction)

        # --- Drawing (for visualization during generation) ---
        screen.fill(BLACK)
        draw_road(screen)
        draw_lane_lines(screen)
        car.draw(screen)
        
        # --- Capture Camera View & Determine Label ---
        camera_view_array, camera_rect = get_camera_view(screen, car)

        # --- real-time camera view display ---
        # Convert the normalized NumPy array back to a Pygame Surface for display
        display_img_array = (camera_view_array * 255).astype(np.uint8)

        # Check if the image array is grayscale (2D) or color (3D)
        if len(display_img_array.shape) == 2: # This means it's grayscale (H, W)
            # For grayscale, pygame.surfarray.make_surface expects (W, H)
            display_img_array_transposed = np.transpose(display_img_array, (1, 0)) 
            camera_display_surf = pygame.surfarray.make_surface(display_img_array_transposed)
        elif len(display_img_array.shape) == 3: # This means it's RGB (H, W, C)
            # For RGB, pygame.surfarray.make_surface expects (W, H, C)
            display_img_array_transposed = np.transpose(display_img_array, (1, 0, 2))
            camera_display_surf = pygame.surfarray.make_surface(display_img_array_transposed)
        else:
            raise ValueError("Unsupported image array shape for display. Expected 2D (H,W) or 3D (H,W,C).")

        # Scale it down for the top-left corner display
        display_width = CAMERA_WIDTH // 2
        display_height = CAMERA_HEIGHT // 2
        scaled_camera_surf = pygame.transform.scale(camera_display_surf, (display_width, display_height))

        # Draw it on the screen (e.g., in the top-left corner)
        screen.blit(scaled_camera_surf, (10, 10))
        # --- END OF ADDED BLOCK ---




        # --- Automated Labeling ---
        # This is the core of "ground truth". For a straight road, the ideal steering angle is 0.
        # If the car is off-center, we calculate an angle to bring it back.

        # Simple approach: Calculate offset from lane center and map to a steering angle
        lane_center_x = SCREEN_WIDTH / 2
        # Offset: positive if car is to the right of center, negative if to the left
        horizontal_offset = car.x - lane_center_x 

        # Convert offset to a 'steering angle' label. This is a crucial mapping.
        # A larger offset means a larger correction angle. You'll need to tune the factor.
        # Example: 1 degree correction for every 5 pixels off center.
        # Higher factor = more aggressive steering
        steering_label = -horizontal_offset * 0.1 # Tune this factor (e.g., 0.1)

        # --- Save Sample ---
        image_filename = f"frame_{samples_generated:05d}.png" # e.g., frame_00000.png
        image_filepath = os.path.join(DATA_DIR, IMAGES_SUBDIR, image_filename)
        #print(f"Attempting to save image to: {image_filepath}")
    
        # Save the image (convert normalized float array back to uint8 for Pillow)
        img_to_save = Image.fromarray((camera_view_array * 255).astype(np.uint8), mode='L') # 'RGB' for colour img, L for grayscale
        img_to_save.save(image_filepath)

        # Append label to CSV
        with open(labels_filepath, 'a') as f:
            f.write(f"{image_filename},{steering_label}\n")

        samples_generated += 1

        # Display progress (optional)
        if samples_generated % 100 == 0:
            print(f"Generated {samples_generated}/{num_samples} samples.")

        # Update display for visualization
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print(f"Data generation complete. Saved {samples_generated} samples to {DATA_DIR}")
         
#-------------------------------------------------------Main Execution Block:
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Data Generation Simulator")
    clock = pygame.time.Clock()

    # Initialize car near the bottom, pointing up
    initial_car_x = SCREEN_WIDTH / 2
    initial_car_y = SCREEN_HEIGHT - CAR_HEIGHT - 50 
    car = Car(initial_car_x, initial_car_y)

    generate_straight_road_data(screen, clock, car, NUM_SAMPLES)


