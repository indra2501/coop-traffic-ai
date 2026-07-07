import pygame
import time
import math
import numpy as np
import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

# Check for PyTorch and PyG availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch or PyG not available. Running in simplified mode.")
    TORCH_AVAILABLE = False

pygame.init()

# -------------------------
# Display config
# -------------------------
SQUARE_SIZE = 800
PANEL_WIDTH = 400
WIDTH = PANEL_WIDTH + SQUARE_SIZE
HEIGHT = SQUARE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Indian Traffic System: Optimized 20-Second Signal Timing")

# -------------------------
# Colors & fonts
# -------------------------
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GROUND = (144, 238, 144)
GROUND_ALT = (134, 228, 134)
LEAVES = (34, 139, 34)
TRUNK = (101, 67, 33)
BLACK = (0, 0, 0)
SHADOW = (20, 20, 20, 100)
RED = (255, 0, 0)
AMBER = (255, 165, 0)
GREEN = (0, 255, 0)
SIDEWALK = (130, 130, 130)
PANEL_BG = (90, 90, 90)
GRASS_GREEN = (144, 238, 144)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
ROUGH = (80, 80, 80)
WET = (0, 160, 255)
RAIN_COLOR = (0, 120, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
DARK_GREEN = (0, 100, 0)
LIGHT_GRAY = (180, 180, 180)
DARK_RED = (139, 0, 0)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 139)

font = pygame.font.SysFont(None, 28)
font_panel = pygame.font.SysFont(None, 26)
font_large = pygame.font.SysFont(None, 36)
font_small = pygame.font.SysFont(None, 22)

# -------------------------
# Road geometry - INDIAN LEFT-SIDE DRIVING
# -------------------------
lane_width = 60  # Increased from 45 to 60 for wider lanes
road_width = lane_width * 6  # 3 lanes per direction (360 pixels total)
scene_offset = PANEL_WIDTH
road_x = scene_offset + SQUARE_SIZE // 2 - road_width // 2
road_y = SQUARE_SIZE // 2 - road_width // 2

center_x = road_x + road_width // 2
center_y = road_y + road_width // 2

# Stop lines positioned closer to intersection for better flow
stop_line_margin = 10
traffic_boundary_distance = 120  # Reduced for better flow

stop_lines = {
    'north': center_y - traffic_boundary_distance,
    'south': center_y + traffic_boundary_distance, 
    'east': center_x - traffic_boundary_distance,
    'west': center_x + traffic_boundary_distance
}

# -------------------------
# INDIAN LEFT-SIDE DRIVING Lane Configuration
# -------------------------
# VERTICAL LANES (moving up/down):
#   Lane 7: Vertical UP (leftmost)    - x position: road_x + lane_width * 0.5
#   Lane 8: Vertical UP (middle)      - x position: road_x + lane_width * 1.5
#   Lane 9: Vertical UP (rightmost)   - x position: road_x + lane_width * 2.5
#   Lane 1: Vertical DOWN (leftmost)  - x position: road_x + lane_width * 3.5
#   Lane 2: Vertical DOWN (middle)    - x position: road_x + lane_width * 4.5
#   Lane 3: Vertical DOWN (rightmost) - x position: road_x + lane_width * 5.5
#
# HORIZONTAL LANES (INDIAN LEFT-SIDE):
#   RIGHT SIDE of yellow divider (top part) - vehicles moving LEFT to RIGHT
#   LEFT SIDE of yellow divider (bottom part) - vehicles moving RIGHT to LEFT
#
#   Lane 10: Horizontal RIGHT (top)   - y position: road_y + lane_width * 0.5
#   Lane 11: Horizontal RIGHT (middle)- y position: road_y + lane_width * 1.5
#   Lane 12: Horizontal RIGHT (bottom)- y position: road_y + lane_width * 2.5
#   Lane 6: Horizontal LEFT (top)     - y position: road_y + lane_width * 3.5
#   Lane 5: Horizontal LEFT (middle)  - y position: road_y + lane_width * 4.5
#   Lane 4: Horizontal LEFT (bottom)  - y position: road_y + lane_width * 5.5

lane_centers = {
    # Vertical UP lanes (moving upward)
    7: (road_x + lane_width * 0.5, 'vertical', 'up'),
    8: (road_x + lane_width * 1.5, 'vertical', 'up'),
    9: (road_x + lane_width * 2.5, 'vertical', 'up'),
    
    # Vertical DOWN lanes (moving downward)
    1: (road_x + lane_width * 3.5, 'vertical', 'down'),
    2: (road_x + lane_width * 4.5, 'vertical', 'down'),
    3: (road_x + lane_width * 5.5, 'vertical', 'down'),
    
    # Horizontal RIGHT lanes (moving rightward) - INDIAN RIGHT SIDE
    10: (road_y + lane_width * 0.5, 'horizontal', 'right'),
    11: (road_y + lane_width * 1.5, 'horizontal', 'right'),
    12: (road_y + lane_width * 2.5, 'horizontal', 'right'),
    
    # Horizontal LEFT lanes (moving leftward) - INDIAN LEFT SIDE
    6: (road_y + lane_width * 3.5, 'horizontal', 'left'),
    5: (road_y + lane_width * 4.5, 'horizontal', 'left'),
    4: (road_y + lane_width * 5.5, 'horizontal', 'left'),
}

# Legacy arrays for compatibility (will map to new system)
vertical_lane_centers = [
    road_x + lane_width * 0.5,      # Lane 7 - UP (leftmost)
    road_x + lane_width * 1.5,      # Lane 8 - UP 
    road_x + lane_width * 2.5,      # Lane 9 - UP (rightmost)
    road_x + lane_width * 3.5,      # Lane 1 - DOWN (leftmost)
    road_x + lane_width * 4.5,      # Lane 2 - DOWN
    road_x + lane_width * 5.5       # Lane 3 - DOWN (rightmost)
]

horizontal_lane_centers = [
    road_y + lane_width * 0.5,      # Lane 10 - RIGHT (top)
    road_y + lane_width * 1.5,      # Lane 11 - RIGHT (middle)
    road_y + lane_width * 2.5,      # Lane 12 - RIGHT (bottom)
    road_y + lane_width * 3.5,      # Lane 6 - LEFT (top)
    road_y + lane_width * 4.5,      # Lane 5 - LEFT (middle)
    road_y + lane_width * 5.5       # Lane 4 - LEFT (bottom)
]

# Road segments for different conditions
road_segments = [
    {"type": "straight", "start": 0, "end": 150},
    {"type": "potholes", "start": 150, "end": 320},
    {"type": "rough", "start": 320, "end": 470},
    {"type": "wet", "start": 470, "end": SQUARE_SIZE}
]

# ENHANCED: 30-second traffic light timing for better flow
light_cycle_duration = 20  # 30 seconds green for optimal flow
amber_duration = 6  # 6 seconds amber

# -------------------------
# Tree positions (static)
# -------------------------
tree_positions = []

# def generate_trees():
#     """Generate tree positions on sides of roads"""
#     global tree_positions
#     tree_positions = []
    
#     for y in range(0, SQUARE_SIZE, 80):
#         x = road_x - random.randint(30, 50)
#         if x > scene_offset + 20:
#             tree_positions.append({'x': x, 'y': y, 'size': random.randint(15, 25)})
    
#     for y in range(0, SQUARE_SIZE, 80):
#         x = road_x + road_width + random.randint(30, 50)
#         if x < WIDTH - 20:
#             tree_positions.append({'x': x, 'y': y, 'size': random.randint(15, 25)})
    
#     for x in range(scene_offset, WIDTH, 80):
#         y = road_y - random.randint(30, 50)
#         if y > 20:
#             tree_positions.append({'x': x, 'y': y, 'size': random.randint(15, 25)})
    
#     for x in range(scene_offset, WIDTH, 80):
#         y = road_y + road_width + random.randint(30, 50)
#         if y < SQUARE_SIZE - 20:
#             tree_positions.append({'x': x, 'y': y, 'size': random.randint(15, 25)})

vehicle_traces = {}

# -------------------------
# CSV Logging System
# -------------------------
class CSVLogger:
    def __init__(self):
        self.filename = f"traffic_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.headers = [
            'timestamp', 'vehicle_id', 'vehicle_type', 'direction', 'lane', 
            'speed', 'x_pos', 'y_pos', 'wait_time', 'distance_traveled',
            'damage', 'action_taken', 'collision_count', 'v2v_messages',
            'signal_state', 'queue_length', 'throughput'
        ]
        self.create_file()
        
    def create_file(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
    def log_vehicle_data(self, data_dict):
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([data_dict.get(header, '') for header in self.headers])
        except PermissionError:
            # File is locked (probably open in Excel) - skip this log entry
            pass

# -------------------------
# ENHANCED Traffic Light System with 30-second cycles and optimized placement
# -------------------------
class TrafficLightSystem:
    def __init__(self):
        self.start_time = time.time()
        self.vertical_green = True
        self.amber_active = False
        self.phase_change_time = self.start_time
        self.cycle_count = 0
        self.countdown_timer = light_cycle_duration
        
    def update(self):
        current_time = time.time()
        elapsed = current_time - self.phase_change_time
        
        # Update countdown timer
        if self.vertical_green and not self.amber_active:
            self.countdown_timer = max(0, light_cycle_duration - elapsed)
        elif not self.vertical_green and not self.amber_active:
            self.countdown_timer = max(0, light_cycle_duration - elapsed)
        else:
            self.countdown_timer = max(0, amber_duration - elapsed)
        
        if self.vertical_green and not self.amber_active:
            if elapsed >= light_cycle_duration:
                self.amber_active = True
                self.phase_change_time = current_time
        elif self.vertical_green and self.amber_active:
            if elapsed >= amber_duration:
                self.vertical_green = False
                self.amber_active = False
                self.phase_change_time = current_time
                self.cycle_count += 1
        elif not self.vertical_green and not self.amber_active:
            if elapsed >= light_cycle_duration:
                self.amber_active = True
                self.phase_change_time = current_time
        else:
            if elapsed >= amber_duration:
                self.vertical_green = True
                self.amber_active = False
                self.phase_change_time = current_time
                self.cycle_count += 1
                
    def get_light_state(self, direction):
        if direction == 'vertical':
            if self.vertical_green and not self.amber_active:
                return 'green'
            elif self.vertical_green and self.amber_active:
                return 'amber'
            else:
                return 'red'
        else:
            if not self.vertical_green and not self.amber_active:
                return 'green'
            elif not self.vertical_green and self.amber_active:
                return 'amber'
            else:
                return 'red'
                
    def get_time_until_change(self):
        return max(0, self.countdown_timer)

# -------------------------
# Enhanced Performance tracking
# -------------------------
performance_data = {
    'collisions': [],
    'avg_speed_vertical': [],
    'avg_speed_horizontal': [],
    'messages_exchanged': [],
    'damage_levels': [],
    'time_steps': [],
    'fed_updates': [],
    'throughput': [],
    'waiting_times': [],
    'v2v_enabled': True,
    'queue_lengths_vertical': [],
    'queue_lengths_horizontal': [],
    'total_vehicles_passed': 0
}

# -------------------------
# Helper functions
# -------------------------
def get_lane_center_x(lane_idx):
    """Get center x coordinate for vertical lanes (UP/DOWN)"""
    # Lanes 7, 8, 9 are vertical UP lanes (leftmost, middle, rightmost)
    # Lanes 1, 2, 3 are vertical DOWN lanes (leftmost, middle, rightmost)
    if lane_idx == 7:  # UP leftmost
        return road_x + lane_width * 0.5
    elif lane_idx == 8:  # UP middle
        return road_x + lane_width * 1.5
    elif lane_idx == 9:  # UP rightmost
        return road_x + lane_width * 2.5
    elif lane_idx == 1:  # DOWN leftmost
        return road_x + lane_width * 3.5
    elif lane_idx == 2:  # DOWN middle
        return road_x + lane_width * 4.5
    elif lane_idx == 3:  # DOWN rightmost
        return road_x + lane_width * 5.5
    else:
        return road_x + road_width // 2

def get_lane_center_y(lane_idx):
    """Get center y coordinate for horizontal lanes (LEFT/RIGHT)"""
    # Lanes 10, 11, 12 are RIGHT lanes (top, middle, bottom) - INDIAN RIGHT SIDE
    # Lanes 6, 5, 4 are LEFT lanes (top, middle, bottom) - INDIAN LEFT SIDE
    if lane_idx == 10:  # RIGHT top
        return road_y + lane_width * 0.5
    elif lane_idx == 11:  # RIGHT middle
        return road_y + lane_width * 1.5
    elif lane_idx == 12:  # RIGHT bottom
        return road_y + lane_width * 2.5
    elif lane_idx == 6:  # LEFT top
        return road_y + lane_width * 3.5
    elif lane_idx == 5:  # LEFT middle
        return road_y + lane_width * 4.5
    elif lane_idx == 4:  # LEFT bottom
        return road_y + lane_width * 5.5
    else:
        return road_y + road_width // 2

def get_lane_center_position(lane_idx):
    """Get x or y position and direction info for any lane"""
    if lane_idx in [7, 8, 9]:  # Vertical UP
        return get_lane_center_x(lane_idx), 'vertical', 'up'
    elif lane_idx in [1, 2, 3]:  # Vertical DOWN
        return get_lane_center_x(lane_idx), 'vertical', 'down'
    elif lane_idx in [10, 11, 12]:  # Horizontal RIGHT - INDIAN RIGHT SIDE
        return get_lane_center_y(lane_idx), 'horizontal', 'right'
    elif lane_idx in [6, 5, 4]:  # Horizontal LEFT - INDIAN LEFT SIDE
        return get_lane_center_y(lane_idx), 'horizontal', 'left'
    return 0, 'vertical', 'up'

def get_current_road_condition(coord):
    for segment in road_segments:
        if segment["start"] <= coord < segment["end"]:
            return segment["type"]
    return "straight"

# def draw_tree(surface, x, y, size):
#     shadow_surface = pygame.Surface((size * 3, size // 2), pygame.SRCALPHA)
#     pygame.draw.ellipse(shadow_surface, (0, 0, 0, 60), (0, 0, size * 3, size // 2))
#     surface.blit(shadow_surface, (x - size * 1.5, y + size * 1.5))
    
#     trunk_width = size // 3
#     trunk_height = size
#     pygame.draw.rect(surface, TRUNK, (x - trunk_width // 2, y, trunk_width, trunk_height))
    
#     pygame.draw.circle(surface, DARK_GREEN, (x, y - size // 3), int(size * 0.8))
#     pygame.draw.circle(surface, LEAVES, (x - size // 2, y - size // 2), int(size * 0.7))
#     pygame.draw.circle(surface, LEAVES, (x + size // 2, y - size // 2), int(size * 0.7))
#     pygame.draw.circle(surface, LEAVES, (x, y - size), int(size * 0.9))

# def draw_trees(surface):
#     for tree in tree_positions:
#         draw_tree(surface, tree['x'], tree['y'], tree['size'])

def draw_lane_markings_vertical():
    # Center yellow line
    center_x_line = road_x + road_width // 2
    pygame.draw.line(screen, YELLOW, (center_x_line, 0), (center_x_line, SQUARE_SIZE), 4)
    
    # Lane dividers (dashed white lines)
    dividers = [
        road_x + lane_width,
        road_x + lane_width * 2,
        road_x + lane_width * 3,
        road_x + lane_width * 4,
        road_x + lane_width * 5
    ]
    dash_len, gap = 15, 15
    for div_x in dividers:
        y = 0
        while y < SQUARE_SIZE:
            pygame.draw.line(screen, WHITE, (div_x, y), (div_x, y + dash_len), 2)
            y += dash_len + gap
            
    # Directional arrows
    arrow_spacing = 120
    for y in range(100, SQUARE_SIZE - 100, arrow_spacing):
        # Up arrows (left side)
        for i in range(3):
            arrow_x = road_x + lane_width * (i + 0.5)
            draw_arrow(screen, WHITE, (arrow_x, y), 0, 20)
        
        # Down arrows (right side)  
        for i in range(3, 6):
            arrow_x = road_x + lane_width * (i + 0.5)
            draw_arrow(screen, WHITE, (arrow_x, y), 180, 20)

def draw_lane_markings_horizontal():
    # Center yellow line
    center_y_line = road_y + road_width // 2
    pygame.draw.line(screen, YELLOW, (scene_offset, center_y_line), (WIDTH, center_y_line), 4)
    
    # Lane dividers
    dividers = [
        road_y + lane_width,
        road_y + lane_width * 2,
        road_y + lane_width * 3,
        road_y + lane_width * 4,
        road_y + lane_width * 5
    ]
    dash_len, gap = 15, 15
    for div_y in dividers:
        x = scene_offset
        while x < WIDTH:
            pygame.draw.line(screen, WHITE, (x, div_y), (x + dash_len, div_y), 2)
            x += dash_len + gap
            
    # Directional arrows - INDIAN LEFT-SIDE DRIVING
    arrow_spacing = 120
    for x in range(scene_offset + 100, WIDTH - 100, arrow_spacing):
        # RIGHT arrows (top side - INDIAN RIGHT SIDE) - moving RIGHT
        for i in range(3):
            arrow_y = road_y + lane_width * (i + 0.5)
            draw_arrow(screen, WHITE, (x, arrow_y), 90, 20)
        
        # LEFT arrows (bottom side - INDIAN LEFT SIDE) - moving LEFT
        for i in range(3, 6):
            arrow_y = road_y + lane_width * (i + 0.5)
            draw_arrow(screen, WHITE, (x, arrow_y), 270, 20)

def draw_arrow(surface, color, pos, angle, size):
    angle_rad = math.radians(angle)
    end_x = pos[0] + size * math.cos(angle_rad)
    end_y = pos[1] + size * math.sin(angle_rad)
    
    pygame.draw.line(surface, color, pos, (end_x, end_y), 3)
    
    # Arrow head
    arrow_angle1 = math.radians(angle + 150)
    arrow_angle2 = math.radians(angle - 150)
    
    head1_x = end_x + size * 0.5 * math.cos(arrow_angle1)
    head1_y = end_y + size * 0.5 * math.sin(arrow_angle1)
    head2_x = end_x + size * 0.5 * math.cos(arrow_angle2) 
    head2_y = end_y + size * 0.5 * math.sin(arrow_angle2)
    
    pygame.draw.polygon(surface, color, [(end_x, end_y), (head1_x, head1_y), (head2_x, head2_y)])

def draw_stop_lines():
    # North stop line
    pygame.draw.line(screen, WHITE, 
                    (road_x, stop_lines['north']), 
                    (road_x + road_width, stop_lines['north']), 4)
    # South stop line
    pygame.draw.line(screen, WHITE, 
                    (road_x, stop_lines['south']), 
                    (road_x + road_width, stop_lines['south']), 4)
    # East stop line
    pygame.draw.line(screen, WHITE, 
                    (stop_lines['east'], road_y), 
                    (stop_lines['east'], road_y + road_width), 4)
    # West stop line  
    pygame.draw.line(screen, WHITE, 
                    (stop_lines['west'], road_y), 
                    (stop_lines['west'], road_y + road_width), 4)

def draw_traffic_boundary():
    boundary_rect = pygame.Rect(
        center_x - traffic_boundary_distance,
        center_y - traffic_boundary_distance,
        traffic_boundary_distance * 2,
        traffic_boundary_distance * 2
    )
    pygame.draw.rect(screen, BLUE, boundary_rect, 3)

def draw_intersection_markings():
    # Draw pedestrian crossings
    cross_width = 20
    cross_spacing = 10
    
    # North crossing
    for i in range(0, road_width, cross_spacing * 2):
        pygame.draw.line(screen, WHITE, 
                        (road_x + i, stop_lines['north'] - cross_width),
                        (road_x + i + cross_spacing, stop_lines['north'] - cross_width), 3)
    
    # South crossing
    for i in range(0, road_width, cross_spacing * 2):
        pygame.draw.line(screen, WHITE,
                        (road_x + i, stop_lines['south'] + cross_width),
                        (road_x + i + cross_spacing, stop_lines['south'] + cross_width), 3)
    
    # East crossing
    for i in range(0, road_width, cross_spacing * 2):
        pygame.draw.line(screen, WHITE,
                        (stop_lines['east'] + cross_width, road_y + i),
                        (stop_lines['east'] + cross_width, road_y + i + cross_spacing), 3)
    
    # West crossing
    for i in range(0, road_width, cross_spacing * 2):
        pygame.draw.line(screen, WHITE,
                        (stop_lines['west'] - cross_width, road_y + i),
                        (stop_lines['west'] - cross_width, road_y + i + cross_spacing), 3)

def draw_vehicle_traces(surface):
    for vehicle_id, trace in vehicle_traces.items():
        if len(trace) > 1:
            for i in range(len(trace) - 1):
                alpha = int(255 * (i / len(trace)))
                color = (*trace[i][2][:3], alpha)
                
                start_pos = (int(trace[i][0]), int(trace[i][1]))
                end_pos = (int(trace[i+1][0]), int(trace[i+1][1]))
                
                line_surf = pygame.Surface((abs(end_pos[0] - start_pos[0]) + 4, 
                                           abs(end_pos[1] - start_pos[1]) + 4), pygame.SRCALPHA)
                
                local_start = (2, 2)
                local_end = (end_pos[0] - start_pos[0] + 2, end_pos[1] - start_pos[1] + 2)
                
                pygame.draw.line(line_surf, color, local_start, local_end, 2)
                surface.blit(line_surf, (min(start_pos[0], end_pos[0]) - 2, 
                                        min(start_pos[1], end_pos[1]) - 2))

# -------------------------
# Enhanced V2V Communication with Safety Protocols
# -------------------------
class V2VCommunication:
    def __init__(self, communication_range=120):
        self.communication_range = communication_range
        self.message_buffer = []
        self.model_update_buffer = []
        self.max_buffer_size = 200
        self.fed_update_count = 0
        self.safety_override_count = 0
        
    def create_message(self, sender_id, message_type, data, priority=1):
        return {
            'sender_id': sender_id,
            'timestamp': time.time(),
            'type': message_type,
            'data': data,
            'priority': priority,
            'ttl': 5.0,
            'position': data.get('position', (0,0))
        }
    
    def broadcast_message(self, message):
        self.message_buffer.append(message)
        now = time.time()
        self.message_buffer = [m for m in self.message_buffer if now - m['timestamp'] < m['ttl']]
        if len(self.message_buffer) > self.max_buffer_size:
            self.message_buffer = self.message_buffer[-self.max_buffer_size:]
    
    def get_nearby_messages(self, vehicle_pos, vehicle_id):
        now = time.time()
        relevant = []
        for message in self.message_buffer:
            if message['sender_id'] == vehicle_id or now - message['timestamp'] > message['ttl']:
                continue
            msg_pos = message['position']
            dist = math.hypot(vehicle_pos[0] - msg_pos[0], vehicle_pos[1] - msg_pos[1])
            if dist <= self.communication_range:
                relevant.append(message)
        relevant.sort(key=lambda x: (x['priority'], -x['timestamp']), reverse=True)
        return relevant[:10]
    
    def broadcast_model_update(self, sender_id, model_state_dict):
        self.model_update_buffer.append((sender_id, copy.deepcopy(model_state_dict), time.time()))
        self.fed_update_count += 1
        if len(self.model_update_buffer) > self.max_buffer_size:
            self.model_update_buffer = self.model_update_buffer[-self.max_buffer_size:]
    
    def get_nearby_model_updates(self, vehicle_pos, vehicle_id):
        now = time.time()
        updates = []
        for sid, state_dict, ts in self.model_update_buffer:
            if sid == vehicle_id or now - ts > 5.0:
                continue
            updates.append(state_dict)
        return updates
        
    def apply_safety_override(self, vehicle, intended_action, vehicles_list, traffic_light_system):
        """Safety shield that overrides unsafe actions"""
        original_action = intended_action
        
        # Check for red light violation
        light_state = traffic_light_system.get_light_state(vehicle.direction)
        if light_state == 'red' and vehicle.is_at_stop_line() and not vehicle.is_past_stop_line():
            if intended_action in [4, 0]:  # accelerate or maintain
                intended_action = 3  # brake
                self.safety_override_count += 1
        
        # Check for collision with front vehicle
        front_distance = vehicle.get_front_vehicle_distance(vehicles_list)
        if front_distance < vehicle.safe_following_distance * 0.6:
            if intended_action in [4, 0]:  # accelerate or maintain
                intended_action = 3  # brake
                self.safety_override_count += 1
                
        # Check for intersection collision
        if vehicle.is_near_intersection():
            for other in vehicles_list:
                if other.vehicle_id == vehicle.vehicle_id:
                    continue
                if other.is_near_intersection() and not other.is_past_stop_line():
                    distance = math.hypot(vehicle.x - other.x, vehicle.y - other.y)
                    if distance < 50:
                        intended_action = 5  # emergency stop
                        self.safety_override_count += 1
                        break
        
        if original_action != intended_action:
            vehicle.safety_override_active = True
        else:
            vehicle.safety_override_active = False
            
        return intended_action

# -------------------------
# Enhanced GNN + DQN definitions with Federated Learning
# -------------------------
if TORCH_AVAILABLE:
    class TrafficGNN(nn.Module):
        def __init__(self, input_dim=20, hidden_dim=64, output_dim=32):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
            self.conv3 = GCNConv(hidden_dim // 2, output_dim)
            
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x = torch.relu(self.conv3(x, edge_index))
            return x

    class DQNHead(nn.Module):
        def __init__(self, input_dim=32, hidden_dim=128, num_actions=6):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, num_actions)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    def federated_averaging(models):
        if not models:
            return
        avg_state = copy.deepcopy(models[0].state_dict())
        for key in avg_state.keys():
            for model in models[1:]:
                avg_state[key] += model.state_dict()[key]
            avg_state[key] /= len(models)
        for model in models:
            model.load_state_dict(avg_state)

    class ReplayBuffer:
        def __init__(self, capacity=5000):
            self.buffer = deque(maxlen=capacity)
            
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
            
        def sample(self, batch_size):
            if len(self.buffer) < batch_size:
                return None
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return (torch.stack(states),
                    torch.tensor(actions, dtype=torch.long),
                    torch.tensor(rewards, dtype=torch.float),
                    torch.stack(next_states),
                    torch.tensor(dones, dtype=torch.float))
        
        def __len__(self):
            return len(self.buffer)
else:
    class TrafficGNN:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, device):
            return self
        def __call__(self, data):
            return np.zeros((0, 32)) if not hasattr(data, 'x') else np.zeros((data.x.shape[0], 32))
    
    class DQNHead:
        def __init__(self, *args, **kwargs):
            self.state_dict = lambda: {}
            self.load_state_dict = lambda x: None
        def to(self, device):
            return self
    
    def federated_averaging(models):
        pass
    
    class ReplayBuffer:
        def __init__(self, capacity=5000):
            self.buffer = deque(maxlen=capacity)
        def push(self, state, action, reward, next_state, done):
            pass
        def sample(self, batch_size):
            return None, None, None, None, None
        def __len__(self):
            return 0

# ========================= GLOBAL HELPER FUNCTIONS =========================

def determine_turn_intention(lane_idx, traffic_light_system):
    """
    Determine turn intention based on traffic flow state with INDIAN LEFT-SIDE DRIVING.
    
    VERTICAL TRAFFIC FLOW (green):
        - Lane 1 (DOWN-left) → LEFT turn to Lane 4 (LEFT bottom)
        - Lane 2 (DOWN-middle) → STRAIGHT to Lane 8 (UP middle)
        - Lane 3 (DOWN-right) → RIGHT turn to Lane 10 (RIGHT top)
        - Lane 7 (UP-left) → LEFT turn to Lane 12 (RIGHT bottom)
        - Lane 8 (UP-middle) → STRAIGHT to Lane 2 (DOWN middle)
        - Lane 9 (UP-right) → RIGHT turn to Lane 6 (LEFT top)
        
    HORIZONTAL TRAFFIC FLOW (green):
        - Lane 10 (RIGHT-top) → LEFT turn to Lane 7 (UP left)
        - Lane 11 (RIGHT-middle) → STRAIGHT to Lane 5 (LEFT middle)
        - Lane 12 (RIGHT-bottom) → RIGHT turn to Lane 1 (DOWN left)
        - Lane 6 (LEFT-top) → LEFT turn to Lane 9 (UP right)
        - Lane 5 (LEFT-middle) → STRAIGHT to Lane 11 (RIGHT middle)
        - Lane 4 (LEFT-bottom) → RIGHT turn to Lane 3 (DOWN right)
    """
    if traffic_light_system is None:
        return 'straight'
    
    vertical_light = traffic_light_system.get_light_state('vertical')
    horizontal_light = traffic_light_system.get_light_state('horizontal')
    
    if vertical_light == 'green':
        # VERTICAL TRAFFIC FLOW - vertical lanes can proceed
        if lane_idx == 1:  # DOWN leftmost
            return 'left'
        elif lane_idx == 2:  # DOWN middle
            return 'straight'
        elif lane_idx == 3:  # DOWN rightmost
            return 'right'
        elif lane_idx == 7:  # UP leftmost
            return 'left'
        elif lane_idx == 8:  # UP middle
            return 'straight'
        elif lane_idx == 9:  # UP rightmost
            return 'right'
        else:
            return 'straight'
            
    elif horizontal_light == 'green':
        # HORIZONTAL TRAFFIC FLOW - horizontal lanes can proceed
        if lane_idx == 10:  # RIGHT top
            return 'left'
        elif lane_idx == 11:  # RIGHT middle
            return 'straight'
        elif lane_idx == 12:  # RIGHT bottom
            return 'right'
        elif lane_idx == 6:  # LEFT top
            return 'left'
        elif lane_idx == 5:  # LEFT middle
            return 'straight'
        elif lane_idx == 4:  # LEFT bottom
            return 'right'
        else:
            return 'straight'
    else:
        return 'straight'

def calculate_turn_parameters_new(vehicle):
    """
    Calculate precise turn parameters for correct lane-to-lane transitions with INDIAN LEFT-SIDE DRIVING.
    """
    if vehicle.turn_intention == 'straight':
        return False
    
    lane_idx = vehicle.base_lane_idx
    
    # VERTICAL DOWN vehicles (lanes 1, 2, 3)
    if vehicle.direction == 'vertical' and vehicle.movement_direction == 'down':
        if lane_idx == 1 and vehicle.turn_intention == 'left':  # Lane 1 LEFT turn to Lane 4 (LEFT bottom)
            vehicle.turn_target_direction = 'horizontal'
            vehicle.turn_target_movement = 'left'
            vehicle.turn_target_lane = 4
            vehicle.target_visual_angle = 180
            
            # Turn center for left turn from DOWN to LEFT
            turn_offset = 40
            vehicle.turn_center = (center_x - turn_offset, center_y + turn_offset)
            vehicle.turn_start_angle = math.pi  # 180 degrees (pointing left)
            vehicle.turn_end_angle = math.pi / 2  # 90 degrees (pointing down, but in LEFT direction context)
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 2 and vehicle.turn_intention == 'straight':  # Lane 2 STRAIGHT to Lane 8 (UP middle)
            vehicle.turn_target_direction = 'vertical'
            vehicle.turn_target_movement = 'up'
            vehicle.turn_target_lane = 8
            vehicle.target_visual_angle = 270
            
            # Straight through intersection
            turn_offset = 40
            vehicle.turn_center = (center_x, center_y)
            vehicle.turn_start_angle = math.pi  # 180 degrees
            vehicle.turn_end_angle = 0  # 0 degrees (pointing up)
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 3 and vehicle.turn_intention == 'right':  # Lane 3 RIGHT turn to Lane 10 (RIGHT top)
            vehicle.turn_target_direction = 'horizontal'
            vehicle.turn_target_movement = 'right'
            vehicle.turn_target_lane = 10
            vehicle.target_visual_angle = 0
            
            # Turn center for right turn from DOWN to RIGHT
            turn_offset = 40
            vehicle.turn_center = (center_x + turn_offset, center_y + turn_offset)
            vehicle.turn_start_angle = math.pi  # 180 degrees
            vehicle.turn_end_angle = -math.pi / 2  # -90 degrees
            vehicle.turn_radius = 50
            return True
    
    # VERTICAL UP vehicles (lanes 7, 8, 9)
    elif vehicle.direction == 'vertical' and vehicle.movement_direction == 'up':
        if lane_idx == 7 and vehicle.turn_intention == 'left':  # Lane 7 LEFT turn to Lane 12 (RIGHT bottom)
            vehicle.turn_target_direction = 'horizontal'
            vehicle.turn_target_movement = 'right'
            vehicle.turn_target_lane = 12
            vehicle.target_visual_angle = 0
            
            # Turn center for left turn from UP to RIGHT
            turn_offset = 40
            vehicle.turn_center = (center_x + turn_offset, center_y - turn_offset)
            vehicle.turn_start_angle = 0  # 0 degrees (pointing right)
            vehicle.turn_end_angle = -math.pi / 2  # -90 degrees
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 8 and vehicle.turn_intention == 'straight':  # Lane 8 STRAIGHT to Lane 2 (DOWN middle)
            vehicle.turn_target_direction = 'vertical'
            vehicle.turn_target_movement = 'down'
            vehicle.turn_target_lane = 2
            vehicle.target_visual_angle = 90
            
            # Straight through intersection
            turn_offset = 40
            vehicle.turn_center = (center_x, center_y)
            vehicle.turn_start_angle = 0  # 0 degrees
            vehicle.turn_end_angle = math.pi  # 180 degrees
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 9 and vehicle.turn_intention == 'right':  # Lane 9 RIGHT turn to Lane 6 (LEFT top)
            vehicle.turn_target_direction = 'horizontal'
            vehicle.turn_target_movement = 'left'
            vehicle.turn_target_lane = 6
            vehicle.target_visual_angle = 180
            
            # Turn center for right turn from UP to LEFT
            turn_offset = 40
            vehicle.turn_center = (center_x - turn_offset, center_y - turn_offset)
            vehicle.turn_start_angle = 0  # 0 degrees
            vehicle.turn_end_angle = math.pi / 2  # 90 degrees
            vehicle.turn_radius = 50
            return True
    
    # HORIZONTAL RIGHT vehicles (lanes 10, 11, 12)
    elif vehicle.direction == 'horizontal' and vehicle.movement_direction == 'right':
        if lane_idx == 10 and vehicle.turn_intention == 'left':  # Lane 10 LEFT turn to Lane 7 (UP left)
            vehicle.turn_target_direction = 'vertical'
            vehicle.turn_target_movement = 'up'
            vehicle.turn_target_lane = 7
            vehicle.target_visual_angle = 270
            
            # Turn center for left turn from RIGHT to UP
            turn_offset = 40
            vehicle.turn_center = (center_x - turn_offset, center_y - turn_offset)
            vehicle.turn_start_angle = -math.pi / 2  # -90 degrees (pointing up in RIGHT context)
            vehicle.turn_end_angle = 0  # 0 degrees
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 11 and vehicle.turn_intention == 'straight':  # Lane 11 STRAIGHT to Lane 5 (LEFT middle)
            vehicle.turn_target_direction = 'horizontal'
            vehicle.turn_target_movement = 'left'
            vehicle.turn_target_lane = 5
            vehicle.target_visual_angle = 180
            
            # Straight crossing
            turn_offset = 40
            vehicle.turn_center = (center_x, center_y)
            vehicle.turn_start_angle = 0  # 0 degrees (pointing right)
            vehicle.turn_end_angle = math.pi  # 180 degrees (pointing left)
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 12 and vehicle.turn_intention == 'right':  # Lane 12 RIGHT turn to Lane 1 (DOWN left)
            vehicle.turn_target_direction = 'vertical'
            vehicle.turn_target_movement = 'down'
            vehicle.turn_target_lane = 1
            vehicle.target_visual_angle = 90
            
            # Turn center for right turn from RIGHT to DOWN
            turn_offset = 40
            vehicle.turn_center = (center_x - turn_offset, center_y + turn_offset)
            vehicle.turn_start_angle = -math.pi / 2  # -90 degrees
            vehicle.turn_end_angle = math.pi  # 180 degrees
            vehicle.turn_radius = 50
            return True
    
    # HORIZONTAL LEFT vehicles (lanes 6, 5, 4)
    elif vehicle.direction == 'horizontal' and vehicle.movement_direction == 'left':
        if lane_idx == 6 and vehicle.turn_intention == 'left':  # Lane 6 LEFT turn to Lane 9 (UP right)
            vehicle.turn_target_direction = 'vertical'
            vehicle.turn_target_movement = 'up'
            vehicle.turn_target_lane = 9
            vehicle.target_visual_angle = 270
            
            # Turn center for left turn from LEFT to UP
            turn_offset = 40
            vehicle.turn_center = (center_x + turn_offset, center_y - turn_offset)
            vehicle.turn_start_angle = math.pi / 2  # 90 degrees (pointing up in LEFT context)
            vehicle.turn_end_angle = 0  # 0 degrees
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 5 and vehicle.turn_intention == 'straight':  # Lane 5 STRAIGHT to Lane 11 (RIGHT middle)
            vehicle.turn_target_direction = 'horizontal'
            vehicle.turn_target_movement = 'right'
            vehicle.turn_target_lane = 11
            vehicle.target_visual_angle = 0
            
            # Straight crossing
            turn_offset = 40
            vehicle.turn_center = (center_x, center_y)
            vehicle.turn_start_angle = math.pi  # 180 degrees (pointing left)
            vehicle.turn_end_angle = 0  # 0 degrees (pointing right)
            vehicle.turn_radius = 50
            return True
            
        elif lane_idx == 4 and vehicle.turn_intention == 'right':  # Lane 4 RIGHT turn to Lane 3 (DOWN right)
            vehicle.turn_target_direction = 'vertical'
            vehicle.turn_target_movement = 'down'
            vehicle.turn_target_lane = 3
            vehicle.target_visual_angle = 90
            
            # Turn center for right turn from LEFT to DOWN
            turn_offset = 40
            vehicle.turn_center = (center_x + turn_offset, center_y + turn_offset)
            vehicle.turn_start_angle = math.pi / 2  # 90 degrees (pointing down in LEFT context)
            vehicle.turn_end_angle = math.pi  # 180 degrees
            vehicle.turn_radius = 50
            return True
    
    return False

# ========================= ENHANCED VEHICLE CLASS =========================

class BaseVehicle:
    ACTIONS = ['maintain', 'change_left', 'change_right', 'brake', 'accelerate', 'emergency_stop']
    
    def __init__(self, x, y, direction, lane_idx, movement_direction, speed=3, color=RED, vehicle_id=None, width=35, height=18):
        self.x = float(x)
        self.y = float(y)
        self.direction = direction  # 'vertical' or 'horizontal'
        self.lane_idx = lane_idx
        self.movement_direction = movement_direction  # 'up', 'down', 'left', 'right'
        self.speed = float(speed)
        self.base_speed = float(speed)
        self.color = color
        self.width = width
        self.height = height
        self.vehicle_id = vehicle_id or f"V_{random.randint(1000,9999)}"
        
        self.angle = 0.0
        self.visual_direction = self.get_initial_visual_direction()
        
        self.device = 'cpu'
        if TORCH_AVAILABLE:
            self.dqn_head = DQNHead().to(self.device)
            self.dqn_optimizer = optim.Adam(self.dqn_head.parameters(), lr=1e-4)
            self.replay = ReplayBuffer()
        else:
            self.dqn_head = DQNHead()
            self.replay = ReplayBuffer()
        
        self.received_messages = []
        self.last_broadcast_time = 0.0
        self.broadcast_interval = 0.3
        
        self.target_lane = lane_idx
        self.lane_change_progress = 0.0
        self.is_changing_lanes = False
        self.lane_change_cooldown = 0
        
        self.is_waiting_at_light = False
        self.waiting_time = 0
        self.total_wait_time = 0
        self.collision_prevention_active = False
        self.emergency_brake_active = False
        self.safety_override_active = False
        self.safe_following_distance = 80
        self.brake_distance = 100
        
        # Enhanced turning system based on dynamic traffic flow (1-12)
        # Turn intention will be determined dynamically based on traffic light state
        # This is set initially but can change during traffic flow
        self.base_lane_idx = lane_idx
        self.turn_intention = self.determine_turn_intention(lane_idx, None)  # Will be updated with traffic state
        
        self.is_turning = False
        self.turn_progress = 0.0
        self.turn_start_pos = (0, 0)
        self.turn_center = (0, 0)
        self.turn_radius = 0
        self.turn_start_angle = 0
        self.turn_end_angle = 0
        self.turn_target_direction = None
        self.turn_target_lane = None
        self.turn_target_movement = None
        self.turn_speed_multiplier = 1.0
        self.pre_turn_speed = 0.0
        self.target_visual_angle = 0.0
        
        self.is_in_intersection = False
        self.intersection_entry_time = 0
        self.waiting_for_intersection = False
        
        self.damage = 0
        self.distance_traveled = 0
        self.last_x = self.x
        self.last_y = self.y
        self.last_action = 0
        
        self.gamma = 0.99
        self.epsilon = 0.15
        self.train_batch = 32
        self.local_steps = 0
        self.last_training_step = 0
        self.last_fed_sharing_step = 0  # Track federated learning updates
        self.fed_sharing_interval = 100  # Share model every 100 steps
        
        if self.vehicle_id not in vehicle_traces:
            vehicle_traces[self.vehicle_id] = deque(maxlen=100)

    def get_initial_visual_direction(self):
        if self.movement_direction == 'right':
            return 0
        elif self.movement_direction == 'down':
            return 90
        elif self.movement_direction == 'left':
            return 180
        elif self.movement_direction == 'up':
            return 270
        return 0

    def update_path_trace(self):
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        
        if self.vehicle_id in vehicle_traces:
            vehicle_traces[self.vehicle_id].append((center_x, center_y, self.color))

    def is_near_intersection(self):
        if self.direction == 'vertical':
            if self.movement_direction == 'down':
                return self.y >= (center_y - traffic_boundary_distance - 50) and self.y <= (center_y + traffic_boundary_distance)
            elif self.movement_direction == 'up':
                return self.y <= (center_y + traffic_boundary_distance + 50) and self.y >= (center_y - traffic_boundary_distance)
        else:
            if self.movement_direction == 'right':
                return self.x >= (center_x - traffic_boundary_distance - 50) and self.x <= (center_x + traffic_boundary_distance)
            elif self.movement_direction == 'left':
                return self.x <= (center_x + traffic_boundary_distance + 50) and self.x >= (center_x - traffic_boundary_distance)
        return False

    def is_at_stop_line(self):
        stop_margin = 15
        
        if self.direction == 'vertical':
            if self.movement_direction == 'up':
                return abs(self.y + self.height - stop_lines['north']) < stop_margin
            else:  # down
                return abs(self.y - stop_lines['south']) < stop_margin
        else:  # horizontal
            if self.movement_direction == 'right':
                return abs(self.x - stop_lines['east']) < stop_margin
            else:  # left
                return abs(self.x + self.width - stop_lines['west']) < stop_margin

    def determine_turn_intention(self, lane_idx, traffic_light_system):
        """Wrapper to call global determine_turn_intention function"""
        return determine_turn_intention(lane_idx, traffic_light_system)

    def is_past_stop_line(self):
        if self.direction == 'vertical':
            if self.movement_direction == 'up':
                return self.y + self.height < stop_lines['north']
            else:  # down
                return self.y > stop_lines['south']
        else:  # horizontal
            if self.movement_direction == 'right':
                return self.x > stop_lines['east']
            else:  # left
                return self.x + self.width < stop_lines['west']

    def get_sensor_data(self, vehicles_list, traffic_light_system):
        sensor = np.zeros(20, dtype=np.float32)
        
        # Basic vehicle state
        sensor[0] = self.speed / 8.0
        sensor[1] = self.damage / 100.0
        sensor[2] = self.lane_idx / 12.0  # 12 lanes total
        sensor[3] = 1.0 if self.is_changing_lanes else 0.0
        sensor[4] = 1.0 if self.is_turning else 0.0
        
        # Road conditions
        road_coord = self.y if self.direction == 'vertical' else self.x
        cond = get_current_road_condition(road_coord)
        sensor[5] = 1.0 if cond == 'potholes' else 0.0
        sensor[6] = 1.0 if cond == 'wet' else 0.0
        sensor[7] = 1.0 if cond == 'rough' else 0.0
        
        # Traffic light state
        light_state = traffic_light_system.get_light_state(self.direction)
        sensor[8] = 1.0 if light_state == 'red' else 0.0
        sensor[9] = 1.0 if light_state == 'amber' else 0.0
        sensor[10] = 1.0 if light_state == 'green' else 0.0
        
        # Distance to intersection
        dist_to_intersection = self.get_distance_to_intersection()
        sensor[11] = min(dist_to_intersection / 200.0, 1.0)
        
        # Vehicle proximity
        front_distance = self.get_front_vehicle_distance(vehicles_list)
        sensor[12] = min(front_distance / 150.0, 1.0)
        
        left_occupied, right_occupied = self.check_lane_occupancy(vehicles_list)
        sensor[13] = 1.0 if left_occupied else 0.0
        sensor[14] = 1.0 if right_occupied else 0.0
        
        # Emergency states
        sensor[15] = 1.0 if self.emergency_brake_active else 0.0
        sensor[16] = 1.0 if self.safety_override_active else 0.0
        
        # Turn intentions
        if self.turn_intention == 'left':
            sensor[17] = 1.0
            sensor[18] = 0.0
        elif self.turn_intention == 'right':
            sensor[17] = 0.0
            sensor[18] = 1.0
        else:
            sensor[17] = 0.0
            sensor[18] = 0.0
            
        # Waiting state
        sensor[19] = 1.0 if self.is_waiting_at_light else 0.0
        
        if TORCH_AVAILABLE:
            return torch.tensor(sensor, dtype=torch.float, device=self.device)
        else:
            return sensor

    def get_front_vehicle_distance(self, vehicles_list):
        min_distance = 300.0
        
        for other in vehicles_list:
            if other.vehicle_id == self.vehicle_id:
                continue
            
            # Only consider vehicles in same direction and lane
            if (other.direction != self.direction or 
                other.lane_idx != self.lane_idx or 
                other.movement_direction != self.movement_direction):
                continue
            
            if self.movement_direction == 'up':
                if other.y < self.y and abs(other.x - self.x) < self.width:
                    distance = self.y - (other.y + other.height)
                    min_distance = min(min_distance, max(0, distance))
            elif self.movement_direction == 'down':
                if other.y > self.y and abs(other.x - self.x) < self.width:
                    distance = other.y - (self.y + self.height)
                    min_distance = min(min_distance, max(0, distance))
            elif self.movement_direction == 'left':
                if other.x < self.x and abs(other.y - self.y) < self.height:
                    distance = self.x - (other.x + other.width)
                    min_distance = min(min_distance, max(0, distance))
            elif self.movement_direction == 'right':
                if other.x > self.x and abs(other.y - self.y) < self.height:
                    distance = other.x - (self.x + self.width)
                    min_distance = min(min_distance, max(0, distance))
        
        return min_distance

    def check_lane_occupancy(self, vehicles_list):
        left_occupied = False
        right_occupied = False
        
        if self.direction == 'vertical':
            # For vertical roads, left is smaller lane index, right is larger
            left_lane = self.lane_idx - 1 if self.lane_idx > 0 else None
            right_lane = self.lane_idx + 1 if self.lane_idx < 5 else None
        else:
            # For horizontal roads, left is smaller lane index, right is larger  
            left_lane = self.lane_idx - 1 if self.lane_idx > 0 else None
            right_lane = self.lane_idx + 1 if self.lane_idx < 5 else None
        
        for other in vehicles_list:
            if (other.vehicle_id == self.vehicle_id or 
                other.direction != self.direction or
                other.movement_direction != self.movement_direction):
                continue
            
            distance = math.hypot(self.x - other.x, self.y - other.y)
            
            if distance < 80:  # Only consider nearby vehicles
                if left_lane is not None and other.lane_idx == left_lane:
                    left_occupied = True
                if right_lane is not None and other.lane_idx == right_lane:
                    right_occupied = True
        
        return left_occupied, right_occupied

    def can_enter_intersection(self, vehicles_list, traffic_light_system):
        if self.is_in_intersection:
            return True
            
        light_state = traffic_light_system.get_light_state(self.direction)
        if light_state != 'green':
            return False
            
        # Check for vehicles already in intersection
        for other in vehicles_list:
            if other.vehicle_id == self.vehicle_id:
                continue
                
            if other.is_in_intersection:
                distance = math.hypot(self.x - other.x, self.y - other.y)
                if distance < 80:
                    return False
                    
        return True

    def select_action(self, node_embedding, vehicles_list, traffic_light_system):
        front_distance = self.get_front_vehicle_distance(vehicles_list)
        
        # Emergency brake if too close to front vehicle
        if front_distance < self.safe_following_distance * 0.5:
            self.emergency_brake_active = True
            return 5
        
        self.emergency_brake_active = False
        
        light_state = traffic_light_system.get_light_state(self.direction)
        
        # Stop at red light before stop line
        if light_state == 'red' and self.is_near_intersection() and not self.is_past_stop_line():
            dist_to_stop = self.get_distance_to_stop_line()
            if dist_to_stop < 50:
                return 3
        
        # Caution at amber light
        if light_state == 'amber' and self.is_near_intersection() and not self.is_past_stop_line():
            dist_to_stop = self.get_distance_to_stop_line()
            if dist_to_stop < 60:
                return 3
        
        # Wait if intersection is occupied
        if self.is_near_intersection() and not self.is_in_intersection:
            if not self.can_enter_intersection(vehicles_list, traffic_light_system):
                self.waiting_for_intersection = True
                return 3
            else:
                self.waiting_for_intersection = False
        
        # AI decision making
        if TORCH_AVAILABLE and random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.dqn_head(node_embedding.unsqueeze(0))
                return int(q_values.argmax(dim=1).item())
        else:
            # Rule-based fallback
            if front_distance < self.brake_distance:
                return 3
            elif self.speed < self.base_speed and front_distance > self.safe_following_distance * 1.5:
                return 4
            else:
                return 0

    def get_distance_to_stop_line(self):
        if self.direction == 'vertical':
            if self.movement_direction == 'up':
                return abs(self.y + self.height - stop_lines['north'])
            else:  # down
                return abs(self.y - stop_lines['south'])
        else:  # horizontal
            if self.movement_direction == 'right':
                return abs(self.x - stop_lines['east'])
            else:  # left
                return abs(self.x + self.width - stop_lines['west'])

    def get_distance_to_intersection(self):
        return math.hypot(self.x - center_x, self.y - center_y)

    def apply_action(self, action_idx, vehicles_list, traffic_light_system):
        self.last_action = action_idx
        
        # Safety check: never accelerate when too close to front vehicle
        front_distance = self.get_front_vehicle_distance(vehicles_list)
        if front_distance < self.safe_following_distance * 0.7 and action_idx == 4:
            action_idx = 3
        
        light_state = traffic_light_system.get_light_state(self.direction)
        
        # ENHANCED: More efficient traffic flow with 30-second cycles
        # Only stop if very close to stop line during red/amber
        if light_state == 'red' and self.is_at_stop_line() and not self.is_past_stop_line():
            dist_to_stop = self.get_distance_to_stop_line()
            if dist_to_stop < 30:  # Only stop when very close
                self.speed = max(0, self.speed - 2.0)
                self.is_waiting_at_light = True
                self.waiting_time += 1
                self.total_wait_time += 1
                return
        
        # ENHANCED: More permissive amber light behavior
        if light_state == 'amber' and self.is_at_stop_line() and not self.is_past_stop_line():
            dist_to_stop = self.get_distance_to_stop_line()
            if dist_to_stop < 40:  # Only slow down when close
                self.speed = max(1.0, self.speed - 0.5)  # Gentle braking
                self.is_waiting_at_light = True
                self.waiting_time += 1
                self.total_wait_time += 1
                return
        
        if self.waiting_for_intersection and self.is_near_intersection():
            # ENHANCED: More efficient intersection entry
            self.speed = max(1.0, self.speed - 1.0)  # Gentle slowing
            self.is_waiting_at_light = True
            self.waiting_time += 1
            self.total_wait_time += 1
            return
        else:
            self.is_waiting_at_light = False
            self.waiting_time = 0
        
        # Lane change cooldown
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1
        
        # Execute action
        if action_idx == 1 and not self.is_changing_lanes and self.lane_change_cooldown == 0:
            new_lane = self.get_safe_left_lane()
            if new_lane is not None:
                self.initiate_lane_change(new_lane)
        elif action_idx == 2 and not self.is_changing_lanes and self.lane_change_cooldown == 0:
            new_lane = self.get_safe_right_lane()
            if new_lane is not None:
                self.initiate_lane_change(new_lane)
        elif action_idx == 3:  # brake
            self.speed = max(0.5, self.speed - 1.2)
        elif action_idx == 4:  # accelerate
            max_speed = 6.0 if isinstance(self, Car) else 4.0
            self.speed = min(max_speed, self.speed + 0.8)
        elif action_idx == 5:  # emergency stop
            self.speed = max(0, self.speed - 2.5)

    def get_safe_left_lane(self):
        """Get left lane according to Indian left-side driving - PREVENTS crossing center divider"""
        # Vertical lanes: UP (7,8,9) and DOWN (1,2,3)
        # Horizontal lanes: LEFT (6,5,4) and RIGHT (10,11,12)
        
        if self.direction == 'vertical':
            if self.movement_direction == 'down':
                # DOWN lanes: 1, 2, 3 (can only change within these)
                if self.lane_idx == 2:
                    return 1  # Move from center to left
                elif self.lane_idx == 3:
                    return 2  # Move from right to center
            elif self.movement_direction == 'up':
                # UP lanes: 7, 8, 9 (can only change within these)
                if self.lane_idx == 8:
                    return 7  # Move from center to left
                elif self.lane_idx == 9:
                    return 8  # Move from right to center
        else:  # horizontal
            if self.movement_direction == 'right':
                # RIGHT lanes: 10, 11, 12 (can only change within these)
                if self.lane_idx == 11:
                    return 10  # Move from center to left (top)
                elif self.lane_idx == 12:
                    return 11  # Move from bottom to center
            elif self.movement_direction == 'left':
                # LEFT lanes: 6, 5, 4 (can only change within these)
                if self.lane_idx == 5:
                    return 6  # Move from center to left (top)
                elif self.lane_idx == 4:
                    return 5  # Move from bottom to center
        return None

    def get_safe_right_lane(self):
        """Get right lane according to Indian left-side driving - PREVENTS crossing center divider"""
        # Vertical lanes: UP (7,8,9) and DOWN (1,2,3)
        # Horizontal lanes: LEFT (6,5,4) and RIGHT (10,11,12)
        
        if self.direction == 'vertical':
            if self.movement_direction == 'down':
                # DOWN lanes: 1, 2, 3 (can only change within these)
                if self.lane_idx == 1:
                    return 2  # Move from left to center
                elif self.lane_idx == 2:
                    return 3  # Move from center to right
            elif self.movement_direction == 'up':
                # UP lanes: 7, 8, 9 (can only change within these)
                if self.lane_idx == 7:
                    return 8  # Move from left to center
                elif self.lane_idx == 8:
                    return 9  # Move from center to right
        else:  # horizontal
            if self.movement_direction == 'right':
                # RIGHT lanes: 10, 11, 12 (can only change within these)
                if self.lane_idx == 10:
                    return 11  # Move from top to center
                elif self.lane_idx == 11:
                    return 12  # Move from center to bottom
            elif self.movement_direction == 'left':
                # LEFT lanes: 6, 5, 4 (can only change within these)
                if self.lane_idx == 6:
                    return 5  # Move from top to center
                elif self.lane_idx == 5:
                    return 4  # Move from center to bottom
        return None

    def initiate_lane_change(self, target_lane):
        self.target_lane = target_lane
        self.is_changing_lanes = True
        self.lane_change_progress = 0.0
        self.lane_change_cooldown = 60

    def update_lane_change(self):
        if self.is_changing_lanes:
            self.lane_change_progress += 0.02
            
            if self.lane_change_progress >= 1.0:
                self.lane_idx = self.target_lane
                self.is_changing_lanes = False
                self.lane_change_progress = 0.0
            else:
                # Smooth lane change interpolation
                if self.direction == 'vertical':
                    start_x = get_lane_center_x(self.lane_idx) - self.width // 2
                    end_x = get_lane_center_x(self.target_lane) - self.width // 2
                    self.x = start_x + (end_x - start_x) * self.lane_change_progress
                else:
                    start_y = get_lane_center_y(self.lane_idx) - self.height // 2
                    end_y = get_lane_center_y(self.target_lane) - self.height // 2
                    self.y = start_y + (end_y - start_y) * self.lane_change_progress

    def should_turn(self, traffic_light_system):
        if not self.is_near_intersection():
            return False
            
        light_state = traffic_light_system.get_light_state(self.direction)
        if light_state != 'green' or self.is_past_stop_line():
            return False
            
        dist_to_center = self.get_distance_to_intersection()
        return dist_to_center < traffic_boundary_distance and self.turn_intention != 'straight'

    def calculate_turn_parameters(self):
        """Wrapper to call global calculate_turn_parameters_new function"""
        return calculate_turn_parameters_new(self)
    
    def start_turning(self):
        if self.is_turning or self.turn_intention == 'straight':
            return
        
        if not self.calculate_turn_parameters():
            return
        
        self.is_turning = True
        self.turn_progress = 0.0
        self.turn_start_pos = (self.x, self.y)
        self.is_in_intersection = True
        
        self.pre_turn_speed = self.speed
        # Adjust speed for turning
        if self.turn_intention == 'left':
            self.turn_speed_multiplier = 0.60
        else:
            self.turn_speed_multiplier = 0.75
        
        self.speed = self.speed * self.turn_speed_multiplier
    
    def update_turning(self):
        if not self.is_turning:
            return
        
        # Progress the turn
        turn_increment = 0.025
        self.turn_progress += turn_increment
        
        # Gradually recover speed after 70% of turn
        if self.turn_progress > 0.7:
            recovery_rate = (self.turn_progress - 0.7) / 0.3
            target_speed = self.pre_turn_speed * (self.turn_speed_multiplier + (1.0 - self.turn_speed_multiplier) * recovery_rate)
            self.speed = self.speed * 0.95 + target_speed * 0.05
        
        # Complete the turn
        if self.turn_progress >= 1.0:
            self.is_turning = False
            self.turn_progress = 0.0
            self.direction = self.turn_target_direction
            self.lane_idx = self.turn_target_lane
            self.movement_direction = self.turn_target_movement
            self.is_in_intersection = False
            self.angle = self.target_visual_angle
            self.visual_direction = self.target_visual_angle
            self.speed = self.pre_turn_speed
            
            # Snap exactly to target lane center
            if self.direction == 'vertical':
                self.x = get_lane_center_x(self.lane_idx) - self.width / 2
            else:
                self.y = get_lane_center_y(self.lane_idx) - self.height / 2
            return
        
        # Calculate position along turn arc
        current_angle = self.turn_start_angle + (self.turn_end_angle - self.turn_start_angle) * self.turn_progress
        
        self.x = self.turn_center[0] + self.turn_radius * math.cos(current_angle) - self.width / 2
        self.y = self.turn_center[1] + self.turn_radius * math.sin(current_angle) - self.height / 2
        
        # Smooth visual rotation
        angle_diff = self.target_visual_angle - self.visual_direction
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        rotation_speed = 3.0
        if abs(angle_diff) > rotation_speed:
            self.visual_direction += math.copysign(rotation_speed, angle_diff)
        else:
            self.visual_direction = self.target_visual_angle
        
        # Normalize angle
        while self.visual_direction >= 360:
            self.visual_direction -= 360
        while self.visual_direction < 0:
            self.visual_direction += 360
        
        self.angle = self.visual_direction

    def update_physics(self, traffic_light_system, vehicles_list):
        self.update_path_trace()
        
        # Update turn intention based on current traffic flow
        self.turn_intention = self.determine_turn_intention(self.base_lane_idx, traffic_light_system)
        
        # Emergency vehicles bypass red lights
        is_emergency = isinstance(self, EmergencyVehicle) if 'EmergencyVehicle' in globals() else False
        
        # Check traffic light state for this vehicle's direction
        light_state = traffic_light_system.get_light_state(self.direction)
        
        # Diagonal traffic signal control:
        # VERTICAL FLOW (Top to Bottom & Bottom to Top):
        #   - TOP-LEFT signal controls vehicles going UP (bottom to top) - stop BEFORE entering intersection from bottom
        #   - BOTTOM-RIGHT signal controls vehicles going DOWN (top to bottom) - stop BEFORE entering intersection from top
        # HORIZONTAL FLOW (Left to Right & Right to Left):
        #   - BOTTOM-LEFT signal controls vehicles going LEFT (right to left) - stop away from intersection
        #   - TOP-RIGHT signal controls vehicles going RIGHT (left to right) - stop away from intersection
        
        should_stop = False
        at_stop_line = False
        
        # Only non-emergency vehicles must stop at red/amber lights
        if not is_emergency and (light_state == 'red' or light_state == 'amber'):
            if self.direction == 'vertical':
                if self.movement_direction == 'up':
                    # Controlled by TOP-LEFT signal
                    # Vehicles going UP stop BEFORE entering intersection (from bottom side)
                    stop_line_y = road_y + road_width + 80  # Stop below intersection
                    if self.y >= stop_line_y - 30:
                        should_stop = True
                        at_stop_line = True
                        self.y = stop_line_y
                else:  # down
                    # Controlled by BOTTOM-RIGHT signal
                    # Vehicles going DOWN stop BEFORE entering intersection (from top side)
                    stop_line_y = road_y - 80  # Stop above intersection
                    if self.y <= stop_line_y + 30:
                        should_stop = True
                        at_stop_line = True
                        self.y = stop_line_y
            else:  # horizontal
                if self.movement_direction == 'left':
                    # Controlled by BOTTOM-LEFT signal
                    # Vehicles going LEFT (right to left) stop FAR RIGHT, away from intersection
                    stop_line_x = road_x + road_width + 80  # Stop well to the right
                    if self.x >= stop_line_x - 30:
                        should_stop = True
                        at_stop_line = True
                        self.x = stop_line_x
                else:  # right
                    # Controlled by TOP-RIGHT signal
                    # Vehicles going RIGHT (left to right) stop FAR LEFT, away from intersection
                    stop_line_x = road_x - 80  # Stop well to the left
                    if self.x <= stop_line_x + 30:
                        should_stop = True
                        at_stop_line = True
                        self.x = stop_line_x
        
        if should_stop:
            # COMPLETE STOP - vehicle must remain stopped for entire red light duration
            self.speed = 0
            self.is_waiting_at_light = True
            
            # If at stop line, prevent ANY movement
            if at_stop_line:
                self.last_x = self.x
                self.last_y = self.y
                # Skip all movement processing - return early
                return
        else:
            # Green light or not at intersection - allow movement
            if self.speed < self.base_speed:
                self.speed = min(self.base_speed, self.speed + 0.3)
            self.is_waiting_at_light = False
        
        # Start turning if appropriate
        if self.should_turn(traffic_light_system) and self.can_enter_intersection(vehicles_list, traffic_light_system):
            self.start_turning()
        
        if self.is_turning:
            self.update_turning()
            return
        
        self.update_lane_change()
        
        # Maintain lane position when not changing lanes or turning
        if not self.is_changing_lanes and not self.is_turning:
            if self.direction == 'vertical':
                target_x = get_lane_center_x(self.lane_idx) - self.width // 2
                # Smooth position correction
                correction_speed = 0.1
                self.x += (target_x - self.x) * correction_speed
            else:
                target_y = get_lane_center_y(self.lane_idx) - self.height // 2
                correction_speed = 0.1
                self.y += (target_y - self.y) * correction_speed
        
        # Apply road condition effects
        coord = self.y if self.direction == 'vertical' else self.x
        condition = get_current_road_condition(coord)
        
        if condition == "potholes":
            self.speed = max(1.0, self.speed - 0.3)
            if random.random() < 0.003:
                self.damage = min(100, self.damage + 1)
        elif condition == "rough":
            self.speed = max(1.5, self.speed - 0.15)
        elif condition == "wet":
            self.speed = max(2.0, self.speed - 0.1)
            if random.random() < 0.02:
                # Slight skidding on wet roads
                if self.direction == 'vertical':
                    self.x += random.uniform(-0.3, 0.3)
                else:
                    self.y += random.uniform(-0.3, 0.3)
        
        # Update position based on movement direction
        if self.direction == 'vertical':
            if self.movement_direction == 'down':
                self.y += self.speed
                if self.y > SQUARE_SIZE:
                    self.y = -self.height
                    self.damage = max(0, self.damage - 2)
                    performance_data['total_vehicles_passed'] += 1
            else:  # up
                self.y -= self.speed
                if self.y < -self.height:
                    self.y = SQUARE_SIZE + self.height
                    self.damage = max(0, self.damage - 2)
                    performance_data['total_vehicles_passed'] += 1
        else:  # horizontal
            if self.movement_direction == 'right':
                self.x += self.speed
                if self.x > WIDTH:
                    self.x = scene_offset - self.width
                    self.damage = max(0, self.damage - 2)
                    performance_data['total_vehicles_passed'] += 1
            else:  # left
                self.x -= self.speed
                if self.x < scene_offset - self.width:
                    self.x = WIDTH + self.width
                    self.damage = max(0, self.damage - 2)
                    performance_data['total_vehicles_passed'] += 1
        
        # Track distance traveled
        distance_delta = abs(self.x - self.last_x) + abs(self.y - self.last_y)
        self.distance_traveled += distance_delta
        self.last_x = self.x
        self.last_y = self.y

    def process_v2v_messages(self, v2v_system, traffic_light_system):
        if not performance_data['v2v_enabled']:
            return
        
        pos = (self.x + self.width / 2, self.y + self.height / 2)
        
        current_time = time.time()
        if current_time - self.last_broadcast_time > self.broadcast_interval:
            self.broadcast_status(v2v_system, traffic_light_system)
            self.last_broadcast_time = current_time
        
        # Smooth position correction when not turning or changing lanes
        if not self.is_turning and not self.is_changing_lanes:
            if self.direction == 'vertical':
                target_x = get_lane_center_x(self.lane_idx) - self.width / 2
                correction_speed = 0.15
                self.x += (target_x - self.x) * correction_speed
            else:
                target_y = get_lane_center_y(self.lane_idx) - self.height / 2
                correction_speed = 0.15
                self.y += (target_y - self.y) * correction_speed
        
        self.received_messages = v2v_system.get_nearby_messages(pos, self.vehicle_id)
        
        # Process received messages for hazard awareness
        for message in self.received_messages:
            if message['type'] == 'hazard_warning':
                hazard_pos = message['data']['position']
                hazard_type = message['data']['hazard_type']
                distance = math.hypot(self.x - hazard_pos[0], self.y - hazard_pos[1])
                
                if distance < 100:
                    if hazard_type == 'potholes':
                        self.speed = max(1.0, self.speed - 0.5)
                    elif hazard_type in ['wet', 'rough']:
                        self.speed = max(2.0, self.speed - 0.3)
            
            elif message['type'] == 'emergency_brake':
                brake_pos = message['data']['position']
                distance = math.hypot(self.x - brake_pos[0], self.y - brake_pos[1])
                
                if distance < 80:
                    self.speed = max(1.0, self.speed - 1.0)

    def broadcast_status(self, v2v_system, traffic_light_system):
        status = {
            'position': (self.x, self.y),
            'speed': float(self.speed),
            'direction': self.direction,
            'movement_direction': self.movement_direction,
            'lane': int(self.lane_idx),
            'damage': int(self.damage),
            'type': self.__class__.__name__,
            'turn_intention': self.turn_intention,
            'is_turning': self.is_turning,
            'light_state': traffic_light_system.get_light_state(self.direction)
        }
        message = v2v_system.create_message(
            self.vehicle_id, 'status_update', status, priority=1
        )
        v2v_system.broadcast_message(message)
        
        # Broadcast road hazards
        coord = self.y if self.direction == 'vertical' else self.x
        condition = get_current_road_condition(coord)
        if condition in ['potholes', 'wet', 'rough']:
            hazard_data = {
                'position': (self.x, self.y),
                'hazard_type': condition,
                'severity': 'high' if condition == 'potholes' else 'medium'
            }
            hazard_message = v2v_system.create_message(
                self.vehicle_id, 'hazard_warning', hazard_data, priority=3
            )
            v2v_system.broadcast_message(hazard_message)
            
        # Broadcast emergency situations
        if self.emergency_brake_active:
            brake_data = {
                'position': (self.x, self.y),
                'reason': 'emergency_brake',
                'speed': float(self.speed)
            }
            brake_message = v2v_system.create_message(
                self.vehicle_id, 'emergency_brake', brake_data, priority=5
            )
            v2v_system.broadcast_message(brake_message)

    def train_dqn(self):
        if not TORCH_AVAILABLE or len(self.replay) < self.train_batch:
            return
        
        states, actions, rewards, next_states, dones = self.replay.sample(self.train_batch)
        if states is None:
            return
        
        device = next(self.dqn_head.parameters()).device
        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        current_q_values = self.dqn_head(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.dqn_head(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)
        
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_head.parameters(), max_norm=1.0)
        self.dqn_optimizer.step()
        
        self.last_training_step = self.local_steps

    def update_ai_step(self, vehicles_list, v2v_system, node_embedding, traffic_light_system, global_step, logger):
        # Process V2V communications
        self.process_v2v_messages(v2v_system, traffic_light_system)
        
        if TORCH_AVAILABLE:
            state = node_embedding.detach().cpu()
        else:
            state = node_embedding
        
        # Select and apply action with safety override
        intended_action = self.select_action(node_embedding, vehicles_list, traffic_light_system)
        safe_action = v2v_system.apply_safety_override(self, intended_action, vehicles_list, traffic_light_system)
        self.apply_action(safe_action, vehicles_list, traffic_light_system)
        
        self.update_physics(traffic_light_system, vehicles_list)
        
        # Calculate reward and train
        reward = self.calculate_reward(vehicles_list, traffic_light_system)
        
        if TORCH_AVAILABLE:
            next_state = state
            done = 0.0
            self.replay.push(state, safe_action, reward, next_state, done)
        
        self.local_steps += 1
        
        # Train periodically
        if self.local_steps % 10 == 0 and self.local_steps - self.last_training_step >= 10:
            self.train_dqn()
        
        # Federated learning: Share and aggregate models periodically
        if TORCH_AVAILABLE and self.local_steps - self.last_fed_sharing_step >= self.fed_sharing_interval:
            # Broadcast own model to nearby vehicles
            v2v_system.broadcast_model_update(self.vehicle_id, self.dqn_head.state_dict())
            self.last_fed_sharing_step = self.local_steps
            
            # Get nearby model updates and perform federated averaging
            pos = (self.x + self.width / 2, self.y + self.height / 2)
            nearby_updates = v2v_system.get_nearby_model_updates(pos, self.vehicle_id)
            
            # Need at least 2 models to perform meaningful averaging
            if len(nearby_updates) >= 2:
                # Create temporary models from received state dicts
                models_to_average = [self.dqn_head]
                for state_dict in nearby_updates[:4]:  # Limit to 4 nearby models
                    temp_model = self._create_temp_model(state_dict)
                    if temp_model is not None:
                        models_to_average.append(temp_model)
                
                # Perform federated averaging
                if len(models_to_average) > 1:
                    federated_averaging(models_to_average)
        
        # Decay exploration rate
        self.epsilon = max(0.05, self.epsilon * 0.999)
        
        # Log vehicle data
        logger.log_vehicle_data({
            'timestamp': global_step,
            'vehicle_id': self.vehicle_id,
            'vehicle_type': self.__class__.__name__,
            'direction': self.direction,
            'lane': self.lane_idx,
            'speed': self.speed,
            'x_pos': self.x,
            'y_pos': self.y,
            'wait_time': self.total_wait_time,
            'distance_traveled': self.distance_traveled,
            'damage': self.damage,
            'action_taken': safe_action,
            'collision_count': 0,  # Will be updated by collision prevention system
            'v2v_messages': len(self.received_messages),
            'signal_state': traffic_light_system.get_light_state(self.direction),
            'queue_length': len([v for v in vehicles_list if v.is_waiting_at_light]),
            'throughput': performance_data['total_vehicles_passed']
        })
    
    def _create_temp_model(self, state_dict):
        """Create temporary model from state dict for federated averaging."""
        if not TORCH_AVAILABLE:
            return None
        try:
            temp_model = DQNHead(input_dim=32, hidden_dim=128, num_actions=6).to(self.device)
            temp_model.load_state_dict(state_dict)
            return temp_model
        except Exception as e:
            # If model loading fails, skip this update
            return None

    def calculate_reward(self, vehicles_list, traffic_light_system=None):
        reward = 0.0
        
        # Speed reward (higher speed is better but within limits)
        reward += self.speed / 8.0 * 0.5
        
        # Damage penalty
        reward -= self.damage / 100.0 * 0.3
        
        # Safe following distance reward
        front_distance = self.get_front_vehicle_distance(vehicles_list)
        if front_distance > self.safe_following_distance:
            reward += 0.5
        elif front_distance < 30:
            reward -= 2.0
        
        # Lane keeping reward
        if not self.is_changing_lanes:
            reward += 0.3
        
        # Emergency brake penalty
        if not self.emergency_brake_active:
            reward += 0.2
            
        # Turning efficiency reward
        if self.is_turning and self.turn_progress > 0.5:
            reward += 0.3
        
        # Intersection efficiency reward
        if self.is_in_intersection and self.speed > 2.0:
            reward += 0.2
        
        # Speed maintenance reward
        if abs(self.speed - self.base_speed) < 1.0:
            reward += 0.2
            
        # Traffic light compliance reward
        if self.is_waiting_at_light and traffic_light_system is not None:
            if not self.can_enter_intersection(vehicles_list, traffic_light_system):
                reward += 0.1
                
        # V2V communication bonus
        if self.received_messages:
            reward += 0.1 * min(len(self.received_messages), 5)
        
        return reward

    def draw(self, surface):
        damage_ratio = min(self.damage / 100.0, 1.0)
        r, g, b = self.color
        damaged_color = (
            max(0, int(r * (1 - damage_ratio * 0.5))),
            max(0, int(g * (1 - damage_ratio * 0.5))),
            max(0, int(b * (1 - damage_ratio * 0.5)))
        )
        
        # Draw vehicle with rotation if turning
        if abs(self.angle) > 0.5:
            max_dim = max(self.width, self.height) * 2.5
            car_surf = pygame.Surface((int(max_dim), int(max_dim)), pygame.SRCALPHA)
            
            car_center_x = max_dim // 2
            car_center_y = max_dim // 2
            
            car_rect = pygame.Rect(
                int(car_center_x - self.width / 2),
                int(car_center_y - self.height / 2),
                int(self.width),
                int(self.height)
            )
            pygame.draw.rect(car_surf, damaged_color, car_rect, border_radius=3)
            pygame.draw.rect(car_surf, BLACK, car_rect, 2, border_radius=3)
            
            windshield_color = (180, 180, 220)
            windshield_rect = (
                int(car_center_x - self.width / 2 + 3),
                int(car_center_y - self.height / 2 + 3),
                int(self.width - 6),
                int(self.height * 0.25)
            )
            pygame.draw.rect(car_surf, windshield_color, windshield_rect)
            
            rotated_surf = pygame.transform.rotate(car_surf, -self.angle)
            rotated_rect = rotated_surf.get_rect(
                center=(int(self.x + self.width/2), int(self.y + self.height/2))
            )
            
            surface.blit(rotated_surf, rotated_rect.topleft)
        else:
            pygame.draw.rect(surface, damaged_color, 
                            (int(self.x), int(self.y), int(self.width), int(self.height)),
                            border_radius=3)
            
            pygame.draw.rect(surface, BLACK, 
                            (int(self.x), int(self.y), int(self.width), int(self.height)), 
                            2, border_radius=3)
            
            windshield_color = (180, 180, 220)
            if self.direction == 'vertical':
                windshield_rect = (int(self.x + 3), int(self.y + 3), 
                                 int(self.width - 6), int(self.height * 0.25))
            else:
                windshield_rect = (int(self.x + 3), int(self.y + 3), 
                                 int(self.width * 0.25), int(self.height - 6))
            pygame.draw.rect(surface, windshield_color, windshield_rect)
        
        # Status indicators
        indicator_x = int(self.x + self.width // 2)
        indicator_y = int(self.y - 8)
        
        if self.received_messages and performance_data['v2v_enabled']:
            pygame.draw.circle(surface, CYAN, (indicator_x - 8, indicator_y), 3)
        
        if self.emergency_brake_active:
            pygame.draw.circle(surface, RED, (indicator_x, indicator_y), 3)
        
        if self.is_changing_lanes:
            pygame.draw.circle(surface, YELLOW, (indicator_x + 8, indicator_y), 3)
        
        if self.is_waiting_at_light:
            pygame.draw.circle(surface, ORANGE, (indicator_x, indicator_y - 8), 3)
            
        if self.safety_override_active:
            pygame.draw.circle(surface, PURPLE, (indicator_x - 8, indicator_y - 8), 3)
            
        # Turn signals
        if self.is_near_intersection() and not self.is_turning:
            if self.turn_intention == 'left':
                pygame.draw.polygon(surface, GREEN, [
                    (indicator_x - 15, indicator_y - 5),
                    (indicator_x - 5, indicator_y - 5),
                    (indicator_x - 10, indicator_y - 15)
                ])
            elif self.turn_intention == 'right':
                pygame.draw.polygon(surface, GREEN, [
                    (indicator_x + 15, indicator_y - 5),
                    (indicator_x + 5, indicator_y - 5),
                    (indicator_x + 10, indicator_y - 15)
                ])

# -------------------------
# Specific Vehicle Classes
# -------------------------
class Car(BaseVehicle):
    def __init__(self, x, y, direction, lane_idx, movement_direction, speed=4, color=RED, car_id=None):
        super().__init__(x, y, direction, lane_idx, movement_direction, speed, color, car_id, 35, 18)

class Motorbike(BaseVehicle):
    def __init__(self, x, y, direction, lane_idx, movement_direction, speed=5, color=CYAN, bike_id=None):
        super().__init__(x, y, direction, lane_idx, movement_direction, speed, color, bike_id, 25, 12)
        self.safe_following_distance = 40
        self.brake_distance = 60
        
    def draw(self, surface):
        super().draw(surface)
        # Draw rider
        rider_color = BLACK
        rider_x = int(self.x + self.width // 2)
        rider_y = int(self.y + self.height // 2)
        pygame.draw.circle(surface, rider_color, (rider_x, rider_y), 4)

class EmergencyVehicle(BaseVehicle):
    def __init__(self, x, y, direction, lane_idx, movement_direction, speed=6, color=ORANGE, vehicle_id=None):
        super().__init__(x, y, direction, lane_idx, movement_direction, speed, color, vehicle_id, 40, 20)
        self.siren_active = True
        self.siren_timer = 0
        self.priority_override = True
        self.safe_following_distance = 60
        #Reward adjustments
    def apply_action(self, action_idx, vehicles_list, traffic_light_system):
        front_distance = self.get_front_vehicle_distance(vehicles_list)
        if front_distance < self.safe_following_distance * 0.9:
            action_idx = 5
        
        if action_idx == 1 and not self.is_changing_lanes and self.lane_change_cooldown == 0:
            new_lane = self.get_safe_left_lane()
            if new_lane is not None:
                self.initiate_lane_change(new_lane)
        elif action_idx == 2 and not self.is_changing_lanes and self.lane_change_cooldown == 0:
            new_lane = self.get_safe_right_lane()
            if new_lane is not None:
                self.initiate_lane_change(new_lane)
        elif action_idx == 3:
            self.speed = max(2.0, self.speed - 1.0)
        elif action_idx == 4:
            self.speed = min(8.0, self.speed + 1.0)
        elif action_idx == 5:
            self.speed = max(1.0, self.speed - 2.0)
    
    def update_siren(self):
        self.siren_timer += 1
        if self.siren_timer >= 15:
            self.siren_timer = 0
    
    def draw(self, surface):
        super().draw(surface)
        self.update_siren()
        light_color = RED if (self.siren_timer // 5) % 2 == 0 else BLUE
        light_rect = (int(self.x + self.width // 2 - 4), int(self.y - 6), 8, 4)
        pygame.draw.rect(surface, light_color, light_rect)

class Bus(BaseVehicle):
    def __init__(self, x, y, direction, lane_idx, movement_direction, speed=3, color=DARK_BLUE, vehicle_id=None):
        super().__init__(x, y, direction, lane_idx, movement_direction, speed, color, vehicle_id, 45, 25)
        self.safe_following_distance = 100
        self.brake_distance = 120
        
    def draw(self, surface):
        damage_ratio = min(self.damage / 100.0, 1.0)
        r, g, b = self.color
        damaged_color = (
            max(0, int(r * (1 - damage_ratio * 0.5))),
            max(0, int(g * (1 - damage_ratio * 0.5))),
            max(0, int(b * (1 - damage_ratio * 0.5)))
        )
        
        pygame.draw.rect(surface, damaged_color, 
                        (int(self.x), int(self.y), int(self.width), int(self.height)),
                        border_radius=5)
        pygame.draw.rect(surface, BLACK, 
                        (int(self.x), int(self.y), int(self.width), int(self.height)), 
                        2, border_radius=5)
        
        # Windows
        window_color = LIGHT_BLUE
        for i in range(3):
            window_x = self.x + 5 + i * 12
            window_rect = (int(window_x), int(self.y + 5), 10, 8)
            pygame.draw.rect(surface, window_color, window_rect)

# -------------------------
# Enhanced Collision Prevention System
# -------------------------
def prevent_collisions(vehicles):
    collision_count = 0
    
    for i, vehicle1 in enumerate(vehicles):
        for j, vehicle2 in enumerate(vehicles):
            if i >= j:
                continue
            
            # Calculate actual distance between vehicle centers
            center1_x = vehicle1.x + vehicle1.width / 2
            center1_y = vehicle1.y + vehicle1.height / 2
            center2_x = vehicle2.x + vehicle2.width / 2
            center2_y = vehicle2.y + vehicle2.height / 2
            
            distance = math.hypot(center1_x - center2_x, center1_y - center2_y)
            
            # Collision thresholds based on vehicle sizes
            warning_threshold = (vehicle1.width + vehicle2.width) / 2 + 10
            critical_threshold = (vehicle1.width + vehicle2.width) / 2
            
            if distance < warning_threshold:
                rel_speed = abs(vehicle1.speed - vehicle2.speed)
                
                if distance < critical_threshold:
                    collision_count += 1
                    separate_vehicles(vehicle1, vehicle2, critical_threshold - distance)
                    
                    speed_reduction = min(0.5 + rel_speed * 0.2, 1.5)
                    vehicle1.speed = max(2.0, vehicle1.speed - speed_reduction)
                    vehicle2.speed = max(2.0, vehicle2.speed - speed_reduction)
                    
                    if distance < critical_threshold * 0.7:
                        vehicle1.emergency_brake_active = True
                        vehicle2.emergency_brake_active = True
                        
                        damage = int(rel_speed * 0.5)
                        vehicle1.damage = min(100, vehicle1.damage + damage)
                        vehicle2.damage = min(100, vehicle2.damage + damage)
                else:
                    slowdown = (warning_threshold - distance) / warning_threshold
                    vehicle1.speed = max(1.5, vehicle1.speed - slowdown)
                    vehicle2.speed = max(1.5, vehicle2.speed - slowdown)
    
    return collision_count

def separate_vehicles(vehicle1, vehicle2, overlap):
    dx = vehicle1.x - vehicle2.x
    dy = vehicle1.y - vehicle2.y
    distance = math.hypot(dx, dy)
    
    if distance > 0:
        dx /= distance
        dy /= distance
        
        separation_force = overlap * 0.5
        vehicle1.x += dx * separation_force
        vehicle1.y += dy * separation_force
        vehicle2.x -= dx * separation_force
        vehicle2.y -= dy * separation_force
        
        clamp_to_road_bounds(vehicle1)
        clamp_to_road_bounds(vehicle2)

def clamp_to_road_bounds(vehicle):
    if vehicle.direction == 'vertical':
        min_x = road_x + 5
        max_x = road_x + road_width - vehicle.width - 5
        vehicle.x = max(min_x, min(max_x, vehicle.x))
    else:
        min_y = road_y + 5
        max_y = road_y + road_width - vehicle.height - 5
        vehicle.y = max(min_y, min(max_y, vehicle.y))

def build_traffic_graph(vehicles, v2v_system, traffic_light_system, device='cpu'):
    if not vehicles:
        if TORCH_AVAILABLE:
            return Data(x=torch.empty((0, 20)), edge_index=torch.empty((2, 0), dtype=torch.long))
        else:
            class DummyData:
                def __init__(self):
                    self.x = np.empty((0, 20))
                    self.edge_index = [[], []]
            return DummyData()
    
    node_features = []
    vehicle_id_map = {v.vehicle_id: i for i, v in enumerate(vehicles)}
    
    for vehicle in vehicles:
        features = vehicle.get_sensor_data(vehicles, traffic_light_system)
        node_features.append(features)
    
    edge_sources = []
    edge_targets = []
    
    # Physical proximity edges
    for i, vehicle1 in enumerate(vehicles):
        for j, vehicle2 in enumerate(vehicles):
            if i != j:
                distance = math.hypot(vehicle1.x - vehicle2.x, vehicle1.y - vehicle2.y)
                if distance < 150:
                    edge_sources.append(i)
                    edge_targets.append(j)
    
    # V2V communication edges
    for message in v2v_system.message_buffer:
        sender_idx = vehicle_id_map.get(message['sender_id'])
        if sender_idx is not None:
            for receiver_idx in range(len(vehicles)):
                if receiver_idx != sender_idx:
                    receiver = vehicles[receiver_idx]
                    msg_pos = message['position']
                    distance = math.hypot(receiver.x - msg_pos[0], receiver.y - msg_pos[1])
                    if distance < v2v_system.communication_range:
                        edge_sources.append(sender_idx)
                        edge_targets.append(receiver_idx)
    
    if TORCH_AVAILABLE:
        x = torch.stack(node_features).to(device)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long, device=device)
        return Data(x=x, edge_index=edge_index)
    else:
        class DummyData:
            def __init__(self, x, edge_index):
                self.x = x
                self.edge_index = edge_index
        return DummyData(np.array(node_features), [edge_sources, edge_targets])

def update_performance_metrics(vehicles, v2v_system, collision_count, time_step):
    vertical_vehicles = [v for v in vehicles if v.direction == 'vertical']
    horizontal_vehicles = [v for v in vehicles if v.direction == 'horizontal']
    
    avg_speed_vertical = np.mean([v.speed for v in vertical_vehicles]) if vertical_vehicles else 0
    avg_speed_horizontal = np.mean([v.speed for v in horizontal_vehicles]) if horizontal_vehicles else 0
    
    avg_damage = np.mean([v.damage for v in vehicles]) if vehicles else 0
    throughput = performance_data['total_vehicles_passed']
    waiting_vehicles = [v for v in vehicles if v.is_waiting_at_light]
    avg_waiting_time = np.mean([v.waiting_time for v in waiting_vehicles]) if waiting_vehicles else 0
    
    # Queue lengths
    queue_vertical = len([v for v in vertical_vehicles if v.is_waiting_at_light])
    queue_horizontal = len([v for v in horizontal_vehicles if v.is_waiting_at_light])
    
    performance_data['collisions'].append(collision_count)
    performance_data['avg_speed_vertical'].append(avg_speed_vertical)
    performance_data['avg_speed_horizontal'].append(avg_speed_horizontal)
    performance_data['messages_exchanged'].append(len(v2v_system.message_buffer))
    performance_data['damage_levels'].append(avg_damage)
    performance_data['time_steps'].append(time_step)
    performance_data['fed_updates'].append(v2v_system.fed_update_count)
    performance_data['throughput'].append(throughput)
    performance_data['waiting_times'].append(avg_waiting_time)
    performance_data['queue_lengths_vertical'].append(queue_vertical)
    performance_data['queue_lengths_horizontal'].append(queue_horizontal)

def display_performance_graphs():
    if not performance_data['time_steps']:
        print("No performance data to display")
        return
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(performance_data['time_steps'], performance_data['collisions'], 
             label='Collisions', color='red', linewidth=2)
    plt.plot(performance_data['time_steps'], 
             [x/10 for x in performance_data['messages_exchanged']], 
             label='V2V Messages (÷10)', color='blue', linewidth=2)
    plt.title('Safety & Communication Metrics')
    plt.xlabel('Time Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(performance_data['time_steps'], performance_data['avg_speed_vertical'], 
             label='Vertical Avg Speed', color='green', linewidth=2)
    plt.plot(performance_data['time_steps'], performance_data['avg_speed_horizontal'], 
             label='Horizontal Avg Speed', color='purple', linewidth=2)
    plt.title('Average Speed by Direction')
    plt.xlabel('Time Steps')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(performance_data['time_steps'], performance_data['damage_levels'], 
             label='Avg Damage', color='orange', linewidth=2)
    plt.title('Vehicle Damage Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Damage Level')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(performance_data['time_steps'], performance_data['queue_lengths_vertical'], 
             label='Vertical Queue', color='cyan', linewidth=2)
    plt.plot(performance_data['time_steps'], performance_data['queue_lengths_horizontal'], 
             label='Horizontal Queue', color='magenta', linewidth=2)
    plt.title('Queue Lengths at Intersection')
    plt.xlabel('Time Steps')
    plt.ylabel('Queue Length')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(performance_data['time_steps'], performance_data['throughput'], 
             label='Throughput', color='brown', linewidth=2)
    plt.title('System Throughput (Vehicles Completed)')
    plt.xlabel('Time Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(performance_data['time_steps'], performance_data['waiting_times'], 
             label='Avg Waiting Time', color='darkred', linewidth=2)
    plt.title('Average Waiting Time at Lights')
    plt.xlabel('Time Steps')
    plt.ylabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimized_traffic_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Performance graphs saved as 'optimized_traffic_performance.png'")

# -------------------------
# ENHANCED: Optimized Traffic Light Placement for 20-second cycles
# -------------------------
def draw_traffic_lights(surface, traffic_light_system):
    """Draw traffic lights at intersection with diagonal control"""
    # Signal positions and their control assignments:
    # Index 0: Top-left - controls vertical UP (bottom to top)
    # Index 1: Top-right - controls horizontal traffic
    # Index 2: Bottom-left - controls horizontal traffic  
    # Index 3: Bottom-right - controls vertical DOWN (top to bottom)
    
    light_positions = [
        (road_x - 35, road_y - 85),  # Top-left: controls vertical UP
        (road_x + road_width + 15, road_y - 85),  # Top-right: controls horizontal
        (road_x - 50, road_y + road_width - 35),  # Bottom-left: controls horizontal
        (road_x + road_width, road_y + road_width - 35)  # Bottom-right: controls vertical DOWN
    ]
    
    for i, (lx, ly) in enumerate(light_positions):
        # Determine which lights control which traffic
        # Top-left (0) and Bottom-right (3) control VERTICAL traffic
        # Top-right (1) and Bottom-left (2) control HORIZONTAL traffic
        if i == 0 or i == 3:  # Diagonal pair for vertical traffic
            state = traffic_light_system.get_light_state('vertical')
        else:  # i == 1 or i == 2: Diagonal pair for horizontal traffic
            state = traffic_light_system.get_light_state('horizontal')
        
        # Draw light housing
        pygame.draw.rect(surface, BLACK, (lx, ly, 25, 75), border_radius=8)
        
        # Draw individual lights
        red_color = RED if state == 'red' else (80, 0, 0)
        amber_color = AMBER if state == 'amber' else (80, 40, 0)
        green_color = GREEN if state == 'green' else (0, 80, 0)
        
        pygame.draw.circle(surface, red_color, (lx + 12, ly + 15), 10)
        pygame.draw.circle(surface, amber_color, (lx + 12, ly + 37), 10)
        pygame.draw.circle(surface, green_color, (lx + 12, ly + 60), 10)

def draw_road_conditions(surface):
    for segment in road_segments:
        if segment["type"] == "potholes":
            for i in range(5):
                pothole_y = random.randint(segment["start"], segment["end"])
                pothole_x = road_x + random.randint(10, road_width - 20)
                pygame.draw.circle(surface, BLACK, (pothole_x, pothole_y), 8)
        elif segment["type"] == "rough":
            for i in range(15):
                rough_y = random.randint(segment["start"], segment["end"])
                rough_x = road_x + random.randint(0, road_width)
                pygame.draw.circle(surface, ROUGH, (rough_x, rough_y), 3)
        elif segment["type"] == "wet":
            wet_surface = pygame.Surface((road_width, segment["end"] - segment["start"]), pygame.SRCALPHA)
            wet_surface.fill((0, 160, 255, 30))
            surface.blit(wet_surface, (road_x, segment["start"]))

def draw_info_panel(surface, vehicles, v2v_system, traffic_light_system, time_step, fps):
    panel_rect = pygame.Rect(0, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(surface, PANEL_BG, panel_rect)
    pygame.draw.line(surface, WHITE, (PANEL_WIDTH, 0), (PANEL_WIDTH, HEIGHT), 2)
    
    y_offset = 20
    title = font_large.render("Optimized Traffic System", True, WHITE)
    surface.blit(title, (20, y_offset))
    y_offset += 50
    
    # System statistics
    stats = [
        f"Time Step: {time_step}",
        f"FPS: {fps:.1f}",
        f"Vehicles: {len(vehicles)}",
        f"V2V Messages: {len(v2v_system.message_buffer)}",
        f"Fed Updates: {v2v_system.fed_update_count}",
        f"Safety Overrides: {v2v_system.safety_override_count}",
        f"Vehicles Passed: {performance_data['total_vehicles_passed']}",
        "",
        "ENHANCED: 20-Second Signal Timing",
        f"  Vertical: {traffic_light_system.get_light_state('vertical').upper()}",
        f"  Horizontal: {traffic_light_system.get_light_state('horizontal').upper()}",
        f"  Time Left: {traffic_light_system.get_time_until_change():.1f}s",
        "",
        "Traffic Flow Metrics:"
    ]
    
    if vehicles:
        avg_speed = np.mean([v.speed for v in vehicles])
        avg_damage = np.mean([v.damage for v in vehicles])
        waiting_count = sum(1 for v in vehicles if v.is_waiting_at_light)
        turning_count = sum(1 for v in vehicles if v.is_turning)
        avg_wait = np.mean([v.total_wait_time for v in vehicles])
        
        stats.extend([
            f"  Avg Speed: {avg_speed:.2f}",
            f"  Avg Damage: {avg_damage:.1f}",
            f"  Waiting: {waiting_count}",
            f"  Turning: {turning_count}",
            f"  Avg Wait Time: {avg_wait:.1f}",
            f"  Efficiency: {(1 - avg_wait/max(1, time_step)) * 100:.1f}%"
        ])
    
    for line in stats:
        text = font_panel.render(line, True, WHITE)
        surface.blit(text, (20, y_offset))
        y_offset += 30
    
    y_offset += 20
    legend_title = font.render("Vehicle Legend:", True, WHITE)
    surface.blit(legend_title, (20, y_offset))
    y_offset += 35
    
    legend_items = [
        (RED, "Car"),
        (CYAN, "Motorbike"),
        (ORANGE, "Emergency"),
        (DARK_BLUE, "Bus"),
        (GREEN, "Turn Signal"),
        (YELLOW, "Lane Change"),
        (PURPLE, "Safety Override")
    ]
    
    for color, label in legend_items:
        pygame.draw.circle(surface, color, (40, y_offset), 8)
        text = font_panel.render(label, True, WHITE)
        surface.blit(text, (60, y_offset - 12))
        y_offset += 30
    
    y_offset += 20
    controls_title = font.render("Controls:", True, WHITE)
    surface.blit(controls_title, (20, y_offset))
    y_offset += 35
    
    controls = [
        "V - Toggle V2V",
        "P - Performance Graphs",
        "T - Toggle Traces", 
        "R - Reset Simulation",
        "Q - Quit"
    ]
    
    for control in controls:
        text = font_panel.render(control, True, WHITE)
        surface.blit(text, (20, y_offset))
        y_offset += 30

# -------------------------
# ENHANCED: Optimized Vehicle Spawning System for better flow with INDIAN LEFT-SIDE DRIVING
# -------------------------
def spawn_vehicles(vehicles, max_vehicles=15, traffic_light_system=None):
    """
    Spawn vehicles in correct lanes with proper positions for INDIAN LEFT-SIDE DRIVING.
    """
    if len(vehicles) >= max_vehicles:
        return
    
    spawn_probability = 0.04
    
    # Determine which lanes to keep empty based on traffic light state
    vertical_light = traffic_light_system.get_light_state('vertical') if traffic_light_system else None
    horizontal_light = traffic_light_system.get_light_state('horizontal') if traffic_light_system else None
    
    # During vertical flow: keep lanes 4 and 10 empty
    # During horizontal flow: keep lanes 7 and 3 empty
    empty_lanes = []
    if vertical_light == 'green':
        empty_lanes = [4, 10]  # Keep these empty during vertical flow
    elif horizontal_light == 'green':
        empty_lanes = [7, 3]   # Keep these empty during horizontal flow
    
    # Spawn vertical UP vehicles - Lanes 7, 8, 9
    if random.random() < spawn_probability and len(vehicles) < max_vehicles:
        available_lanes = [l for l in [7, 8, 9] if l not in empty_lanes]
        if available_lanes:
            lane = random.choice(available_lanes)
            x = get_lane_center_x(lane) - 17.5  # Center vehicle in lane
            y = SQUARE_SIZE + 50  # Start below screen
            
            # Determine vehicle type
            vehicle_type = random.choices(
                ['car', 'motorbike', 'emergency', 'bus'], 
                weights=[0.6, 0.2, 0.05, 0.15]
            )[0]
            
            if vehicle_type == 'car':
                color = random.choice([RED, BLUE, GREEN, DARK_GREEN, DARK_RED])
                vehicle = Car(x, y, 'vertical', lane, 'up', color=color)
            elif vehicle_type == 'motorbike':
                vehicle = Motorbike(x, y, 'vertical', lane, 'up')
            elif vehicle_type == 'emergency':
                vehicle = EmergencyVehicle(x, y, 'vertical', lane, 'up')
            else:  # bus
                vehicle = Bus(x, y, 'vertical', lane, 'up')
            
            vehicles.append(vehicle)
    
    # Spawn vertical DOWN vehicles - Lanes 1, 2, 3
    if random.random() < spawn_probability and len(vehicles) < max_vehicles:
        available_lanes = [l for l in [1, 2, 3] if l not in empty_lanes]
        if available_lanes:
            lane = random.choice(available_lanes)
            x = get_lane_center_x(lane) - 17.5
            y = -50  # Start above screen
            
            vehicle_type = random.choices(
                ['car', 'motorbike', 'emergency', 'bus'], 
                weights=[0.6, 0.2, 0.05, 0.15]
            )[0]
            
            if vehicle_type == 'car':
                color = random.choice([RED, BLUE, GREEN, DARK_GREEN, DARK_RED])
                vehicle = Car(x, y, 'vertical', lane, 'down', color=color)
            elif vehicle_type == 'motorbike':
                vehicle = Motorbike(x, y, 'vertical', lane, 'down')
            elif vehicle_type == 'emergency':
                vehicle = EmergencyVehicle(x, y, 'vertical', lane, 'down')
            else:  # bus
                vehicle = Bus(x, y, 'vertical', lane, 'down')
            
            vehicles.append(vehicle)
    
    # Spawn horizontal RIGHT vehicles - Lanes 10, 11, 12 (INDIAN RIGHT SIDE)
    if random.random() < spawn_probability and len(vehicles) < max_vehicles:
        available_lanes = [l for l in [10, 11, 12] if l not in empty_lanes]
        if available_lanes:
            lane = random.choice(available_lanes)
            y = get_lane_center_y(lane) - 9  # Center in lane
            x = scene_offset - 50  # Start left of screen
            
            vehicle_type = random.choices(
                ['car', 'motorbike', 'emergency', 'bus'], 
                weights=[0.6, 0.2, 0.05, 0.15]
            )[0]
            
            if vehicle_type == 'car':
                color = random.choice([RED, BLUE, GREEN, DARK_GREEN, DARK_RED])
                vehicle = Car(x, y, 'horizontal', lane, 'right', color=color)
            elif vehicle_type == 'motorbike':
                vehicle = Motorbike(x, y, 'horizontal', lane, 'right')
            elif vehicle_type == 'emergency':
                vehicle = EmergencyVehicle(x, y, 'horizontal', lane, 'right')
            else:  # bus
                vehicle = Bus(x, y, 'horizontal', lane, 'right')
            
            vehicles.append(vehicle)
    
    # Spawn horizontal LEFT vehicles - Lanes 6, 5, 4 (INDIAN LEFT SIDE)
    if random.random() < spawn_probability and len(vehicles) < max_vehicles:
        available_lanes = [l for l in [6, 5, 4] if l not in empty_lanes]
        if available_lanes:
            lane = random.choice(available_lanes)
            y = get_lane_center_y(lane) - 9
            x = WIDTH + 50  # Start right of screen
            
            vehicle_type = random.choices(
                ['car', 'motorbike', 'emergency', 'bus'], 
                weights=[0.6, 0.2, 0.05, 0.15]
            )[0]
            
            if vehicle_type == 'car':
                color = random.choice([RED, BLUE, GREEN, DARK_GREEN, DARK_RED])
                vehicle = Car(x, y, 'horizontal', lane, 'left', color=color)
            elif vehicle_type == 'motorbike':
                vehicle = Motorbike(x, y, 'horizontal', lane, 'left')
            elif vehicle_type == 'emergency':
                vehicle = EmergencyVehicle(x, y, 'horizontal', lane, 'left')
            else:  # bus
                vehicle = Bus(x, y, 'horizontal', lane, 'left')
            
            vehicles.append(vehicle)

def main():
    clock = pygame.time.Clock()
    running = True
    
    # Initialize systems
    vehicles = []
    v2v_system = V2VCommunication()
    traffic_light_system = TrafficLightSystem()
    logger = CSVLogger()
    global_step = 0
    
    # Performance tracking
    fps_history = deque(maxlen=60)
    show_traces = False
    
    # Generate trees
    # generate_trees()
    
    print("Starting Enhanced Traffic Simulation with 20-Second Signal Timing")
    print("Features:")
    print("- INDIAN LEFT-SIDE DRIVING with proper lane assignments")
    print("- 20-second traffic light cycles for optimal flow")
    print("- Enhanced V2V communication with safety protocols")
    print("- Federated learning for distributed AI training")
    print("- Real-time performance monitoring and logging")
    print("- Collision prevention and emergency response")
    
    while running:
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_v:
                    performance_data['v2v_enabled'] = not performance_data['v2v_enabled']
                    print(f"V2V Communication: {'ENABLED' if performance_data['v2v_enabled'] else 'DISABLED'}")
                elif event.key == pygame.K_p:
                    display_performance_graphs()
                elif event.key == pygame.K_t:
                    show_traces = not show_traces
                    print(f"Vehicle Traces: {'SHOWN' if show_traces else 'HIDDEN'}")
                elif event.key == pygame.K_r:
                    # Reset simulation
                    vehicles.clear()
                    vehicle_traces.clear()
                    v2v_system = V2VCommunication()
                    traffic_light_system = TrafficLightSystem()
                    global_step = 0
                    performance_data['total_vehicles_passed'] = 0
                    print("Simulation Reset")
        
        # Update systems
        traffic_light_system.update()
        
        # Spawn vehicles
        spawn_vehicles(vehicles, max_vehicles=15, traffic_light_system=traffic_light_system)
        
        # Build traffic graph for AI decision making
        graph_data = build_traffic_graph(vehicles, v2v_system, traffic_light_system)
        
        if TORCH_AVAILABLE:
            if graph_data.x.shape[0] > 0:
                gnn = TrafficGNN().to('cpu')
                node_embeddings = gnn(graph_data)
            else:
                node_embeddings = torch.empty((0, 32))
        else:
            node_embeddings = np.empty((0, 32))
        
        # Update vehicles with AI
        for i, vehicle in enumerate(vehicles):
            if TORCH_AVAILABLE and graph_data.x.shape[0] > 0:
                vehicle_embedding = node_embeddings[i] if i < node_embeddings.shape[0] else torch.zeros(32)
            else:
                vehicle_embedding = np.zeros(32)
            
            vehicle.update_ai_step(vehicles, v2v_system, vehicle_embedding, 
                                 traffic_light_system, global_step, logger)
        
        # Collision prevention
        collision_count = prevent_collisions(vehicles)
        
        # Remove vehicles that are too damaged
        vehicles = [v for v in vehicles if v.damage < 90]
        
        # Update performance metrics
        update_performance_metrics(vehicles, v2v_system, collision_count, global_step)
        
        # Drawing
        screen.fill(GROUND)
        
        # Draw road
        pygame.draw.rect(screen, GRAY, (road_x, 0, road_width, SQUARE_SIZE))  # Vertical road
        pygame.draw.rect(screen, GRAY, (scene_offset, road_y, SQUARE_SIZE, road_width))  # Horizontal road
        
        # Draw road conditions
        draw_road_conditions(screen)
        
        # Draw lane markings
        draw_lane_markings_vertical()
        draw_lane_markings_horizontal()
        
        # Draw intersection elements
        draw_intersection_markings()
        draw_stop_lines()
        draw_traffic_boundary()
        
        # Draw trees
        # draw_trees(screen)
        
        # Draw vehicle traces
        if show_traces:
            draw_vehicle_traces(screen)
        
        # Draw vehicles
        for vehicle in vehicles:
            vehicle.draw(screen)
        
        # Draw traffic lights
        draw_traffic_lights(screen, traffic_light_system)
        
        # Draw info panel
        current_fps = clock.get_fps()
        fps_history.append(current_fps)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else current_fps
        
        draw_info_panel(screen, vehicles, v2v_system, traffic_light_system, global_step, avg_fps)
        
        pygame.display.flip()
        
        # Cap at 60 FPS
        clock.tick(60)
        global_step += 1
    
    pygame.quit()
    
    # Final performance report
    print("\n=== FINAL PERFORMANCE REPORT ===")
    print(f"Total Time Steps: {global_step}")
    print(f"Total Vehicles Processed: {performance_data['total_vehicles_passed']}")
    print(f"Average Vehicles per Step: {performance_data['total_vehicles_passed'] / max(1, global_step):.3f}")
    print(f"V2V Messages Exchanged: {len(v2v_system.message_buffer)}")
    print(f"Federated Updates: {v2v_system.fed_update_count}")
    print(f"Safety Overrides Applied: {v2v_system.safety_override_count}")
    
    if performance_data['collisions']:
        total_collisions = sum(performance_data['collisions'])
        print(f"Total Collisions Prevented: {total_collisions}")
    
    # Save final performance graphs
    display_performance_graphs()

if __name__ == "__main__":
    main()