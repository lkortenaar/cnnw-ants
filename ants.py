import pygame
import numpy as np
import scipy.ndimage
import matplotlib.cm as cm

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 1000  # Window size
GRID_W, GRID_H = 400, 300 # Simulation grid size (smaller = faster)
SCALE_X = WIDTH / GRID_W
SCALE_Y = HEIGHT / GRID_H

NUM_ANTS = 100000
SENSOR_ANGLE = np.pi / 5  # 45 degrees
SENSOR_DIST = 10           # How far ahead they see
TURN_SPEED = 0.4          # How sharply they turn
RANDOM_STRENGTH = 0.3    # Random wobble (probability factor)
DECAY_RATE = 0.65         # How fast trails vanish
CHAOS_RATE = 0.01   # 1% chance to change direction randomly
DASH_DIST = 10

# --- PREDATOR CONFIGURATION ---
ENABLE_PREDATORS = True    # Toggle predator system
NUM_PREDATORS = 1         # Number of predator agents
PREDATOR_SPEED = 0.5       # Movement speed multiplier
PREDATOR_SENSOR_DIST = 15  # How far predators sense ants
PREDATOR_TURN_SPEED = 0.3  # Turn rate
NEGATIVE_PHEROMONE = -100.0  # Strength of repellent trail
NEGATIVE_DECAY = 0.3       # How fast negative pheromones fade
REPULSION_STRENGTH = 0.8   # How much ants avoid negative pheromones

class AntColony:
    def __init__(self, num_ants, width, height):
        self.width = width
        self.height = height
        self.num_ants = num_ants
        
        # --- 1. CIRCLE INITIALIZATION ---
        
        # A. Settings
        center_x = width / 2
        center_y = height / 2
        # Radius: 40% of the smallest screen dimension
        radius = min(width, height) * 0.4 
        
        # B. Math: Polar -> Cartesian
        # Pick a random angle (0 to 2pi) for every ant
        theta = np.random.rand(num_ants) * 2 * np.pi
        
        # Pick a random distance from center
        # TRICK: np.sqrt() ensures they are spread evenly. 
        # If you want a HOLLOW RING, set this to: r = radius
        r = radius * np.sqrt(np.random.rand(num_ants))
        
        # Convert to X, Y
        self.x = center_x + r * np.cos(theta)
        self.y = center_y + r * np.sin(theta)
        
        # --- 2. Orientation ---
        # Random direction (Chaos)
        self.angle = np.random.rand(num_ants) * 2 * np.pi
        
        # OPTION: Point them OUTWARDS from the start?
        # self.angle = theta 
        
        # OPTION: Point them INWARDS (Implosion)?
        # self.angle = theta + np.pi

        # The Pheromone Grid (Environment)
        self.grid = np.zeros((width, height))

        # --- PREDATOR INITIALIZATION ---
        if ENABLE_PREDATORS:
            self.num_predators = NUM_PREDATORS
            # Spawn predators randomly
            self.pred_x = np.random.rand(NUM_PREDATORS) * width
            self.pred_y = np.random.rand(NUM_PREDATORS) * height
            self.pred_angle = np.random.rand(NUM_PREDATORS) * 2 * np.pi

    # Helper to sample grid safely (wrapping around edges)
    def get_sensor_values(self, x_arr, y_arr):
            # Clip coordinates to grid size
            ix = np.clip(x_arr.astype(int), 0, self.width - 1)
            iy = np.clip(y_arr.astype(int), 0, self.height - 1)
            return self.grid[ix, iy]

    def update_predators(self):
        """Update predator agents - they chase ant pheromones"""
        if not ENABLE_PREDATORS:
            return

        # Sense for positive pheromones (ants)
        cx = self.pred_x + np.cos(self.pred_angle) * PREDATOR_SENSOR_DIST
        cy = self.pred_y + np.sin(self.pred_angle) * PREDATOR_SENSOR_DIST

        lx = self.pred_x + np.cos(self.pred_angle - SENSOR_ANGLE) * PREDATOR_SENSOR_DIST
        ly = self.pred_y + np.sin(self.pred_angle - SENSOR_ANGLE) * PREDATOR_SENSOR_DIST

        rx = self.pred_x + np.cos(self.pred_angle + SENSOR_ANGLE) * PREDATOR_SENSOR_DIST
        ry = self.pred_y + np.sin(self.pred_angle + SENSOR_ANGLE) * PREDATOR_SENSOR_DIST

        # Only look at POSITIVE pheromones (ant trails)
        c_val = np.maximum(0, self.get_sensor_values(cx, cy))
        l_val = np.maximum(0, self.get_sensor_values(lx, ly))
        r_val = np.maximum(0, self.get_sensor_values(rx, ry))

        # Turn toward highest pheromone
        forward_mask = (c_val > l_val) & (c_val > r_val)
        turn_mask = ~forward_mask

        if np.any(turn_mask):
            turn_dir = (r_val[turn_mask] > l_val[turn_mask]).astype(float)
            turn_dir = turn_dir * 2 - 1
            self.pred_angle[turn_mask] += turn_dir * PREDATOR_TURN_SPEED

        # Add some randomness
        self.pred_angle += (np.random.rand(self.num_predators) - 0.5) * 0.2

        # Move
        self.pred_x += np.cos(self.pred_angle) * PREDATOR_SPEED
        self.pred_y += np.sin(self.pred_angle) * PREDATOR_SPEED

        # Wrap
        self.pred_x = self.pred_x % self.width
        self.pred_y = self.pred_y % self.height

        # Deposit NEGATIVE pheromones
        ix = self.pred_x.astype(int)
        iy = self.pred_y.astype(int)
        self.grid[ix, iy] += NEGATIVE_PHEROMONE
    def update(self):
        """The core simulation step."""
        
        # --- A. SENSING (The "Vision Cone") ---
        # We calculate 3 sensor positions for ALL ants simultaneously
        # 1. Center Sensor
        cx = self.x + np.cos(self.angle) * SENSOR_DIST
        cy = self.y + np.sin(self.angle) * SENSOR_DIST
        
        # 2. Left Sensor
        lx = self.x + np.cos(self.angle - SENSOR_ANGLE) * SENSOR_DIST
        ly = self.y + np.sin(self.angle - SENSOR_ANGLE) * SENSOR_DIST
        
        # 3. Right Sensor
        rx = self.x + np.cos(self.angle + SENSOR_ANGLE) * SENSOR_DIST
        ry = self.y + np.sin(self.angle + SENSOR_ANGLE) * SENSOR_DIST

        

        c_val = self.get_sensor_values(cx, cy)
        l_val = self.get_sensor_values(lx, ly)
        r_val = self.get_sensor_values(rx, ry)

        # --- REPULSION FROM NEGATIVE PHEROMONES ---
        if ENABLE_PREDATORS:
            # If sensing negative pheromones, AVOID them
            # Turn away from the most negative direction
            negative_mask = (c_val < 0) | (l_val < 0) | (r_val < 0)

            if np.any(negative_mask):
                # Find which direction is LEAST negative (or most positive)
                # Turn AWAY from the most negative sensor
                repel_left = l_val[negative_mask] < r_val[negative_mask]
                repel_right = ~repel_left

                # Turn away (opposite of attraction)
                repel_dir = np.zeros(np.sum(negative_mask))
                repel_dir[repel_left] = 1  # Turn right to avoid left
                repel_dir[repel_right] = -1  # Turn left to avoid right

                self.angle[negative_mask] += repel_dir * TURN_SPEED * REPULSION_STRENGTH

        # --- B. DECISION LOGIC (Probabilistic + Momentum) ---
        # 1. Forward condition: Center > Left and Center > Right
        # Ant wants to keep moving straight (Momentum)
        forward_mask = (c_val > l_val) & (c_val > r_val)
        
        # 2. Random Steer (The "Probability" factor)
        # Even if pheromones say "go left", we add random noise
        random_steer = (np.random.rand(self.num_ants) - 0.5) * 2 * RANDOM_STRENGTH

        # 3. Turn Logic
        # If Right > Left, turn Right. Else turn Left.
        # We use standard numpy masking to update angles
        turn_mask = ~forward_mask # Ants that need to turn
        
        # Calculate turn direction: +1 (right), -1 (left) based on sensor strength
        # (r_val > l_val) gives True/False, converted to float becomes 1.0/0.0
        turn_dir = (r_val[turn_mask] > l_val[turn_mask]).astype(float) 
        turn_dir = turn_dir * 2 - 1 # Convert to [-1, 1] range
        
        # Apply rotations
        self.angle[turn_mask] += turn_dir * TURN_SPEED
        self.angle += random_steer # Apply noise to everyone

        # --- C. MOVEMENT ---
        # 1. Identify who is bursting this frame
        chaos_roll = np.random.rand(self.num_ants)
        dash_mask = chaos_roll < CHAOS_RATE
        
        # 2. Randomize Angle for bursting ants
        # We give them a totally random direction
        self.angle[dash_mask] = np.random.rand(np.sum(dash_mask)) * 2 * np.pi
        
        # 3. THE KICK: Move them far away instantly
        # Instead of moving 1 step, they move 30 steps in that new direction.
        # This breaks them out of the "pheromone trap" instantly.
        
        # We update x/y for dashing ants differently than normal ants
        # Normal movement (step = 1)
        self.x[~dash_mask] += np.cos(self.angle[~dash_mask])
        self.y[~dash_mask] += np.sin(self.angle[~dash_mask])
        
        # Dash movement (step = 30)
        self.x[dash_mask] += np.cos(self.angle[dash_mask]) * DASH_DIST
        self.y[dash_mask] += np.sin(self.angle[dash_mask]) * DASH_DIST
        
        # Wrap
        self.x = self.x % self.width
        self.y = self.y % self.height

        # Deposit (Keep it to 1 pixel for speed!)
        ix = self.x.astype(int)
        iy = self.y.astype(int)
        self.grid[ix, iy] += 1.0 

    def diffuse(self):
        """The Eulerian step: Blur and Decay the trail map."""
        #clip the grid to avoid overflow
        #self.grid = np.clip(self.grid, 0, 2)
        # 1. Blur (Diffusion) - Spreads pheromones to neighbors
        # This simulates the gas spreading in the air/ground
        self.grid = scipy.ndimage.gaussian_filter(self.grid, sigma=1)
        
        # 2. Decay - Old trails vanish
        if ENABLE_PREDATORS:
            # Positive pheromones decay normally
            positive_mask = self.grid > 0
            self.grid[positive_mask] *= DECAY_RATE

            # Negative pheromones decay faster
            negative_mask = self.grid < 0
            self.grid[negative_mask] *= NEGATIVE_DECAY
        else:
            self.grid *= DECAY_RATE


def main():
    pygame.init()
    # Use HWACCEL for potential speedup
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("Ant Colony Simulation")
    clock = pygame.time.Clock()

    # Initialize Logic
    colony = AntColony(NUM_ANTS, GRID_W, GRID_H)
    
    running = True
    show_visuals = True # Toggle flag

    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    show_visuals = not show_visuals
                    print(f"Visuals: {show_visuals}")
                elif event.key == pygame.K_r:
                    # Reset grid on 'R'
                    colony.grid.fill(0)

        # 2. Simulation Step
        if ENABLE_PREDATORS:
            colony.update_predators()
        colony.update()
        colony.diffuse()

        # 3. Visualization
        # 1. Normalize the grid for the colormap (0.0 to 1.0)
        # Handle both positive and negative values
        if ENABLE_PREDATORS:
            # Split into positive and negative components
            pos_grid = np.maximum(0, colony.grid)
            neg_grid = np.abs(np.minimum(0, colony.grid))

            # Normalize each separately
            max_pos = np.max(pos_grid) + 0.001
            max_neg = np.max(neg_grid) + 0.001

            norm_pos = np.power(pos_grid, 0.5) / np.power(max_pos, 0.5)
            norm_neg = np.power(neg_grid, 0.5) / np.power(max_neg, 0.5)

            # Create color: magma for ants, blue/cyan for predators
            colored_pos = cm.magma(norm_pos)
            colored_neg = np.zeros_like(colored_pos)
            colored_neg[:, :, 2] = norm_neg  # Blue channel for negative
            colored_neg[:, :, 1] = norm_neg * 0.5  # Slight green for cyan

            # Combine (additive)
            colored_grid = np.clip(colored_pos + colored_neg, 0, 1)
        else:
            # Original visualization
            max_val = np.max(colony.grid) + 0.001
            norm_grid = np.power(colony.grid, 0.5) / np.power(max_val, 0.5)
            colored_grid = cm.magma(norm_grid)

        # 3. Convert to 0-255 uint8 for Pygame
        surf_array = (colored_grid[:, :, :3] * 255).astype(np.uint8)

        # 4. Blit
        surface = pygame.surfarray.make_surface(surf_array)
        surface = pygame.transform.scale(surface, (WIDTH, HEIGHT))
        screen.blit(surface, (0, 0))

        pygame.display.flip()

    predator_info = f" | Predators: {NUM_PREDATORS}" if ENABLE_PREDATORS else ""
    pygame.display.set_caption(f"Ants: {NUM_ANTS}{predator_info} | FPS: {clock.get_fps():.1f}")
    clock.tick(60)  # Limit to 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()