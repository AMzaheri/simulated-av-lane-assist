import pygame
import numpy as np

#---------------------------------------- Simulation Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100) # Road color
YELLOW = (255, 255, 0)

# Road parameters
LANE_WIDTH = 125 # Width of a single lane
ROAD_WIDTH = LANE_WIDTH * 2 # Two lanes total
LANE_LINE_WIDTH = 5

# Car parameters
CAR_WIDTH = 30
CAR_HEIGHT = 50
CAR_SPEED = 3 # Pixels per frame
CAR_STEERING_SPEED = 2 # Degrees per frame

# Camera parameters (what the ML model "sees")
CAMERA_WIDTH = 200
CAMERA_HEIGHT = 150
CAMERA_OFFSET_Y = -5 # How far in front of the car the camera is
#CAMERA_Y_RELATIVE_TO_CAR_FRONT = -CAR_HEIGHT / 2 - CAMERA_OFFSET_Y
CAMERA_Y_OFFSET_FROM_CAR_CENTER = -70

# --- ADJUSTED FOR DEBUGGING VISIBILITY: TEST CURVE ---
# This will draw a simple, small, visible arc roughly in the center of your screen.
CURVE_RADIUS = 200 # A smaller, fixed radius for easy visibility
CURVE_CENTER_X = SCREEN_WIDTH / 2 # Center the curve's bounding circle horizontally
CURVE_CENTER_Y = SCREEN_HEIGHT / 2 # Center the curve's bounding circle vertically

# This will draw a 90-degree arc from the top-left to the bottom-left of its bounding circle (a downward left turn).
# It should be clearly visible near the center of your screen.
CURVE_START_ANGLE_DEG = 90  # Start at the top of the circle
CURVE_END_ANGLE_DEG = 180   # End at the left of the circle (clockwise arc)

# Convert angles to radians for numpy/math functions if needed
CURVE_START_ANGLE_RAD = np.radians(CURVE_START_ANGLE_DEG)
CURVE_END_ANGLE_RAD = np.radians(CURVE_END_ANGLE_DEG)

#-----------------------------------------------------Car clasee
# This class will manage the car's position, orientation, and 
#provide methods for movement and drawing.
class Car:
    def __init__(self, x, y, angle=90): # angle=90 means pointing up initially
        self.x = x
        self.y = y
        self.angle = angle # Angle in degrees, 0=right, 90=up, 180=left, 270=down
        self.speed = CAR_SPEED
        # Allow camera offset to be adjusted per car (for diversification)
        self.camera_offset_y = CAMERA_Y_OFFSET_FROM_CAR_CENTER

    def move(self):
        # Convert angle to radians for trigonometric functions
        angle_rad = np.deg2rad(self.angle)
        self.x += self.speed * np.cos(angle_rad)
        self.y -= self.speed * np.sin(angle_rad) # Pygame y-axis is inverted

    def steer(self, direction): # direction: -1 for left, 1 for right
        self.angle += direction * CAR_STEERING_SPEED
        # Keep angle within 0-360 degrees
        self.angle %= 360
   
    # --- Steering in a curved road ---
    def steer_curved_road(self, angle_change_deg):
        """
        Directly applies an angle change to the car's heading.
        Positive angle_change_deg makes the car turn counter-clockwise (left).
        Negative angle_change_deg makes the car turn clockwise (right).
        """
        self.angle += angle_change_deg
        # Keep angle within 0-359 degrees
        self.angle %= 360


    def draw(self, screen):
        # Rotate car image/rectangle
        car_surf = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA) # SRCALPHA for transparency
        pygame.draw.rect(car_surf, (0, 0, 200), (0, 0, CAR_WIDTH, CAR_HEIGHT)) # Blue car
        rotated_car = pygame.transform.rotate(car_surf, self.angle - 90) # Adjust angle for Pygame's default rotation

        # Get the rotated rectangle to position it correctly
        new_rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, new_rect.topleft)

#------------------------------------------------------- Drawing Functions:

def draw_road(screen):
    # Draw the main road rectangle
    road_left_x = SCREEN_WIDTH / 2 - ROAD_WIDTH / 2
    pygame.draw.rect(screen, GRAY, (road_left_x, 0, ROAD_WIDTH, SCREEN_HEIGHT))

def draw_lane_lines(screen):
    road_center_x = SCREEN_WIDTH / 2

    # Left Lane Line (Dashed)
    left_line_x = road_center_x - ROAD_WIDTH / 2 + LANE_WIDTH / 2
    for y in range(0, SCREEN_HEIGHT, LANE_LINE_WIDTH * 3): # Dashed line
        pygame.draw.rect(screen, WHITE, (left_line_x - LANE_LINE_WIDTH / 2, y, LANE_LINE_WIDTH, LANE_LINE_WIDTH * 2))

    # Right Lane Line (Solid Yellow)
    right_line_x = road_center_x + ROAD_WIDTH / 2 - LANE_WIDTH / 2
    pygame.draw.rect(screen, YELLOW, (right_line_x - LANE_LINE_WIDTH / 2, 0, LANE_LINE_WIDTH, SCREEN_HEIGHT))

    # Center Dashed Line (Optional, for 2-way traffic)
    #pygame.draw.rect(screen, WHITE, (road_center_x - LANE_LINE_WIDTH / 2, 0, LANE_LINE_WIDTH, SCREEN_HEIGHT))

#------------------------------------------------Curved Road Drawing Functions

def draw_curved_road(screen, center_x, center_y, radius, start_angle_deg, end_angle_deg, road_width):
    """Draws a curved road segment using thick arcs."""
    # Calculate inner and outer radii for the road surface
    inner_radius = radius - road_width / 2
    outer_radius = radius + road_width / 2

    # Draw the outer edge of the road
    pygame.draw.arc(screen, GRAY,
                    (center_x - outer_radius, center_y - outer_radius, 2 * outer_radius, 2 * outer_radius),
                    np.radians(start_angle_deg), np.radians(end_angle_deg), int(road_width)) # draw as a thick arc


def draw_curved_lane_lines(screen, center_x, center_y, radius, start_angle_deg, end_angle_deg, lane_width, lane_line_width):
    """Draws curved lane lines (yellow solid and white dashed)."""
    
    # Yellow solid line (inner lane marker for a right turn)
    yellow_line_radius = radius - lane_width / 2
    pygame.draw.arc(screen, YELLOW,
                    (center_x - yellow_line_radius, center_y - yellow_line_radius, 2 * yellow_line_radius, 2 * yellow_line_radius),
                    np.radians(start_angle_deg), np.radians(end_angle_deg), int(lane_line_width))

    # White dashed line (outer lane marker for a right turn)
    white_line_radius = radius + lane_width / 2
    
    # For dashed lines on an arc, we'll draw multiple small arcs
    # Define step size for dashes (in degrees)
    dash_step_deg = 5 # Adjust for longer/shorter dashes
    gap_step_deg = 5 # Adjust for longer/shorter gaps

    current_angle_deg = start_angle_deg
    while current_angle_deg < end_angle_deg:
        # Draw a dash
        dash_end_angle_deg = min(current_angle_deg + dash_step_deg, end_angle_deg)
        pygame.draw.arc(screen, WHITE,
                        (center_x - white_line_radius, center_y - white_line_radius, 2 * white_line_radius, 2 * white_line_radius),
                        np.radians(current_angle_deg), np.radians(dash_end_angle_deg), int(lane_line_width))
        
        # Move to the start of the next dash (skipping the gap)
        current_angle_deg += dash_step_deg + gap_step_deg

#------------------------------------------------ Camera View Capture
def get_camera_view(screen, car):
    # Calculate camera top-left position relative to the car's orientation
    # This is simplified. For rotating camera, you'd need more complex geometry.
    # For now, assume camera looks "up" relative to screen, even if car rotates.
    # This means the "camera" is always looking directly up on the screen,
    # which is a common simplification for initial lane-keeping.
    # The ML model then learns from the *orientation of the lane lines* within this fixed view.

    # Calculate the top-left corner of the camera view
    # The camera is always fixed at the top of the car's bounding box
    camera_x = car.x - CAMERA_WIDTH / 2
    #camera_y = car.y + CAMERA_Y_RELATIVE_TO_CAR_FRONT # From the car's "front"
    camera_y = car.y + CAMERA_Y_OFFSET_FROM_CAR_CENTER 

    # Ensure camera view is within screen bounds
    camera_x = max(0, min(camera_x, SCREEN_WIDTH - CAMERA_WIDTH))
    camera_y = max(0, min(camera_y, SCREEN_HEIGHT - CAMERA_HEIGHT))

    # Get the surface rect
    camera_rect = pygame.Rect(camera_x, camera_y, CAMERA_WIDTH, CAMERA_HEIGHT)

    # Capture the surface
    camera_surf = screen.subsurface(camera_rect)

    # Convert to NumPy array
    img_array = pygame.surfarray.array3d(camera_surf)

    # Pygame's array is (width, height, channels) by default, convert to (height, width, channels) for ML
    img_array = np.transpose(img_array, (1, 0, 2))

    # Convert to grayscale for simplicity for the ML model (optional but common)
    # For simplicity, convert directly here. ML model might expect 1 channel.
    # Luminosity method: 0.2989*R + 0.5870*G + 0.1140*B
    grayscale_img = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

    # Normalize to 0-1 range
    normalized_img = grayscale_img / 255.0
    # colour image to test
    ##normalized_img = img_array / 255.0 # Normalize RGB pixel values to 0-1

    return normalized_img, camera_rect # Return the array and the rect for drawing (optional)

#----------------------------------------------Main Simulation Loop
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lane Keeping Simulator")
    clock = pygame.time.Clock()

    # Initialize car in the center of the road, pointing up
    initial_car_x = SCREEN_WIDTH / 2
    initial_car_y = SCREEN_HEIGHT - CAR_HEIGHT - 50 # Start near the bottom
    car = Car(initial_car_x, initial_car_y)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Basic Manual Steering (for testing simulator, will be replaced by AI)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    car.steer(1) # Steer left (adjust angle - simple)
                if event.key == pygame.K_RIGHT:
                    car.steer(-1) # Steer right (adjust angle - simple)

        # --- Update ---
        car.move() # Move the car forward

        # --- Drawing ---
        screen.fill(BLACK) # Clear screen
        draw_road(screen)
        draw_lane_lines(screen)
        car.draw(screen)

        # Capture and display camera view (for debugging)
        camera_view, camera_rect = get_camera_view(screen, car)
        # Resize the camera view for display, if desired (e.g., 2x larger)
        display_camera_view = pygame.transform.scale(pygame.surfarray.make_surface((camera_view * 255).astype(np.uint8)), (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2))
        screen.blit(display_camera_view, (10, 10)) # Display in top-left corner
        pygame.draw.rect(screen, (255,0,0), camera_rect, 1) # Draw red rectangle around captured area

        pygame.display.flip() # Update the full display surface
        clock.tick(FPS) # Limit frame rate

    pygame.quit()
