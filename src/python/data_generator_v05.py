#----------------------------------------------libraries
import pygame
import numpy as np
import os
from PIL import Image

# --- Curve Following Constants ---
LOOK_AHEAD_DISTANCE = 100 # How far ahead (in pixels) the car "looks" on the curve
KP_ANGLE = 0.5            # Proportional gain for angle error (tune this!)
KP_OFFSET = 0.05          # Proportional gain for offset error (tune this!)

# Import simulator components (keep all these imports)
from simulator import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    WHITE, BLACK, GRAY, YELLOW,
    LANE_WIDTH, ROAD_WIDTH, LANE_LINE_WIDTH,
    CAR_WIDTH, CAR_HEIGHT, CAR_SPEED, CAR_STEERING_SPEED,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_OFFSET_Y, CAMERA_Y_OFFSET_FROM_CAR_CENTER,
    Car, draw_road, draw_lane_lines, get_camera_view,
    CURVE_CENTER_X, CURVE_CENTER_Y, CURVE_RADIUS,
    CURVE_START_ANGLE_DEG, CURVE_END_ANGLE_DEG,
    draw_curved_road, draw_curved_lane_lines
)

#-------------------------------------------------------
# --- Data Generation Constants
DATA_DIR = "data"
CURRENT_RUN_NAME = "run_v6_CorrectedCurveMovement" # Changed run name
NUM_SAMPLES = 5000 # Increased samples
ROAD_TYPE = "curved" # Set to "straight" or "curved" here

IMAGES_SUBDIR = os.path.join(DATA_DIR, CURRENT_RUN_NAME, "images")
LABELS_FILE_PATH = os.path.join(DATA_DIR, CURRENT_RUN_NAME, "labels.csv")

#-------------------------------------------------------Automated Driving Logic

def generate_data(screen, clock, car, num_samples, road_type):
    """
    Define how the car "drives" to generate data for various road scenarios,
    based on the specified road_type.
    """
    print(f"Generating {num_samples} samples for {road_type} road...")

    os.makedirs(IMAGES_SUBDIR, exist_ok=True)
    labels_filepath = LABELS_FILE_PATH

    with open(labels_filepath, 'w') as f:
        f.write("image_filename,steering_angle\n")

    samples_generated = 0
    while samples_generated < num_samples:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # --- Update Car Position (Random speed variation can apply to both) ---
        min_speed = CAR_SPEED * 0.8
        max_speed = CAR_SPEED * 1.2
        car.speed = np.random.uniform(min_speed, max_speed)

        # --- Automated Driving Logic & Environment Reset ---
        if road_type == "straight":
            # --- STRAIGHT ROAD LOGIC ---
            # Infinite straight road loop
            if car.y < -CAR_HEIGHT:
                car.y = SCREEN_HEIGHT + CAR_HEIGHT / 2
                car.x = SCREEN_WIDTH / 2 + np.random.uniform(-20, 20) # Reset to center +/- 20 pixels

            # Simulate Car Deviations (random steering)
            if np.random.rand() < 0.15:
                car.steer(np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.4))

            # Steering Corrections for straight road
            safe_left_bound = SCREEN_WIDTH / 2 - ROAD_WIDTH / 2 + CAR_WIDTH / 2
            safe_right_bound = SCREEN_WIDTH / 2 + ROAD_WIDTH / 2 - CAR_WIDTH / 2
            if car.x < safe_left_bound:
                car.steer(4) # Steer right
            elif car.x > safe_right_bound:
                car.steer(-4) # Steer left
            if car.angle > 95: # If car is angled too much to the left (e.g., angle 100), steer right
                car.steer(-2)
            elif car.angle < 85: # If car is angled too much to the right (e.g., angle 80), steer left
                car.steer(2)
            
            steering_label = 0 # Default to 0 for straight unless offset
            lane_center_x = SCREEN_WIDTH / 2
            horizontal_offset = car.x - lane_center_x
            steering_label = -horizontal_offset * 0.1 # Tune this factor to correct for offset

            # --- Environment Reset for straight road (Corrected) ---
            # If car goes too far off the screen, reset it to the bottom
            if car.y < -CAR_HEIGHT or car.y > SCREEN_HEIGHT + CAR_HEIGHT:
                 car.x = SCREEN_WIDTH / 2 + np.random.uniform(-20, 20)
                 car.y = SCREEN_HEIGHT - CAR_HEIGHT - 50
                 car.angle = 90
                 car.camera_offset_y = CAMERA_Y_OFFSET_FROM_CAR_CENTER
            
        elif road_type == "curved":
            # --- CURVED ROAD LOGIC: Pure Pursuit-like Controller ---

            # 1. Calculate car's position relative to the curve's center (for math coordinates)
            car_rel_x = car.x - CURVE_CENTER_X
            car_rel_y_math = -(car.y - CURVE_CENTER_Y) # Flip y-axis for standard math angles (y increases upwards)

            # 2. Calculate the car's current angle relative to the curve's center (polar angle in math coords)
            current_polar_angle_rad = np.arctan2(car_rel_y_math, car_rel_x)

            # 3. Calculate the ideal target point on the curve (look-ahead point)
            angular_step_rad = LOOK_AHEAD_DISTANCE / CURVE_RADIUS
            target_polar_angle_rad = current_polar_angle_rad + angular_step_rad 

            # Calculate the target (x, y) coordinates on the ideal curve (Pygame coordinates)
            target_x = CURVE_CENTER_X + CURVE_RADIUS * np.cos(target_polar_angle_rad)
            target_y = CURVE_CENTER_Y - CURVE_RADIUS * np.sin(target_polar_angle_rad) # Flip y-axis back for Pygame display

            # 4. Calculate the required heading angle for the car to point towards the target
            dx = target_x - car.x
            dy_for_arctan2 = car.y - target_y # This is the crucial change: current_y - target_y for correct angle mapping

            angle_to_target_deg = np.degrees(np.arctan2(dy_for_arctan2, dx))
            # Normalize to 0-360 range
            angle_to_target_deg = (angle_to_target_deg + 360) % 360

            # 5. Calculate the difference between current car angle and desired angle
            angle_error = angle_to_target_deg - car.angle
            # Normalize angle error to -180 to 180 degrees for smoother steering
            angle_error = (angle_error + 180) % 360 - 180

            # 6. Calculate perpendicular distance from car to ideal curve (for offset correction)
            offset_from_ideal_radius = np.sqrt(car_rel_x**2 + car_rel_y_math**2) - CURVE_RADIUS

            # 7. Determine steering label (combining angle error and offset error)
            steering_label = angle_error * KP_ANGLE - offset_from_ideal_radius * KP_OFFSET
            
            # Add a small random component for diversity
            if np.random.rand() < 0.05: # 5% chance of random deviation per frame
                steering_label += np.random.uniform(-0.5, 0.5) # Adjust magnitude as needed

            # Apply steering to the car
            car.steer_curved_road(steering_label)

            # --- Environment Reset for Curved Road ---
            # Calculate current polar angle relative to the curve's center in degrees (0-360 range)
            # This is recalculated here to ensure it's up-to-date after car movement for reset logic.
            current_polar_angle_rad_for_reset = np.arctan2(car_rel_y_math, car_rel_x)
            current_polar_angle_deg_normalized = (np.degrees(current_polar_angle_rad_for_reset) + 360) % 360

            # Reset if car has reached or passed the end of the defined arc (CURVE_END_ANGLE_DEG)
            # OR if it goes significantly off track in other directions.
            if current_polar_angle_deg_normalized >= CURVE_END_ANGLE_DEG or \
               car.x < CURVE_CENTER_X - CURVE_RADIUS - ROAD_WIDTH/2 - CAR_WIDTH/2 or \
               car.y > CURVE_CENTER_Y + ROAD_WIDTH/2 + CAR_HEIGHT/2 or \
               np.sqrt(car_rel_x**2 + car_rel_y_math**2) > CURVE_RADIUS + ROAD_WIDTH/2 + CAR_WIDTH:
                
                car.x = CURVE_CENTER_X
                car.y = CURVE_CENTER_Y - CURVE_RADIUS
                car.angle = 90 # Start pointing up
                car.camera_offset_y = CAMERA_Y_OFFSET_FROM_CAR_CENTER # Reset camera offset

            # Debugging print statements (optional, uncomment to see real-time values)
            # print(f"Car: ({car.x:.1f}, {car.y:.1f}) Angle: {car.angle:.1f} Label: {steering_label:.2f}")
            # print(f"Polar Ang (math): {np.degrees(current_polar_angle_rad_for_reset):.1f}, Target Ang (math): {np.degrees(target_polar_angle_rad):.1f}")
            # print(f"Target Pygame: ({target_x:.1f}, {target_y:.1f})")
            # print(f"dx: {dx:.1f}, dy_for_arctan2: {dy_for_arctan2:.1f}, Angle to Target (Pygame): {angle_to_target_deg:.1f}, Angle Error: {angle_error:.1f}, Offset: {offset_from_ideal_radius:.1f}")

        # Always move the car after determining its steering
        car.move()

        # --- Drawing Road for visualisation ---
        screen.fill(BLACK)
        if road_type == "straight":
            draw_road(screen)
            draw_lane_lines(screen)
        elif road_type == "curved":
            draw_curved_road(screen, CURVE_CENTER_X, CURVE_CENTER_Y, CURVE_RADIUS,
                             CURVE_START_ANGLE_DEG, CURVE_END_ANGLE_DEG, ROAD_WIDTH)
            draw_curved_lane_lines(screen, CURVE_CENTER_X, CURVE_CENTER_Y, CURVE_RADIUS,
                                   CURVE_START_ANGLE_DEG, CURVE_END_ANGLE_DEG, LANE_WIDTH, LANE_LINE_WIDTH)

        car.draw(screen)

        # --- Diversify Camera Position (Applies to both road types) ---
        base_camera_offset_y = CAMERA_Y_OFFSET_FROM_CAR_CENTER
        car.camera_offset_y = base_camera_offset_y + np.random.uniform(-10, 10)

        # --- Capture Camera View & Determine Label (Steering label is set above) ---
        camera_view_array, camera_rect = get_camera_view(screen, car)

        # Only save if car is somewhat on screen (prevents saving black screens when car is off-track)
        if car.x >= -CAR_WIDTH/2 and car.x <= SCREEN_WIDTH + CAR_WIDTH/2 and \
           car.y >= -CAR_HEIGHT/2 and car.y <= SCREEN_HEIGHT + CAR_HEIGHT/2:
            image_filename = f"frame_{samples_generated:05d}.png"
            image_filepath = os.path.join(IMAGES_SUBDIR, image_filename)
            img_to_save = Image.fromarray((camera_view_array * 255).astype(np.uint8), mode='L')
            img_to_save.save(image_filepath)

            with open(labels_filepath, 'a') as f:
                f.write(f"{image_filename},{steering_label}\n")

            samples_generated += 1

            if samples_generated % 100 == 0:
                print(f"Generated {samples_generated}/{num_samples} samples.")
        else:
            # If car goes completely off screen, it means the reset condition probably didn't catch it
            pass

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

    # --- Initialize car based on ROAD_TYPE ---
    if ROAD_TYPE == "straight":
        initial_car_x = SCREEN_WIDTH / 2
        initial_car_y = SCREEN_HEIGHT - CAR_HEIGHT - 50
        initial_car_angle = 90
    elif ROAD_TYPE == "curved":
        # For the test curve (90 to 180 deg, center at SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
        # Car starts at the top of the circle, pointing up (tangent)
        initial_car_x = CURVE_CENTER_X
        initial_car_y = CURVE_CENTER_Y - CURVE_RADIUS
        initial_car_angle = 90 # Tangent at the top of the circle, pointing up

    car = Car(initial_car_x, initial_car_y, angle=initial_car_angle)

    # Pass ROAD_TYPE to the generate_data function
    generate_data(screen, clock, car, NUM_SAMPLES, ROAD_TYPE)
