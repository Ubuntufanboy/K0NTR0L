import numpy as np
import cv2
import av

def create_viz(
    frame: np.ndarray,
    i: int,
    j_left: np.ndarray,
    j_right: np.ndarray,
    buttons: np.ndarray,
    token_set: list,
):
    """
    Visualize gamepad actions alongside a gameplay video frame.
    
    Parameters:
    - frame: Video frame as numpy array
    - i: Current frame index (default 0)
    - j_left: 16x2 array of left joystick positions (-1 to 1)
    - j_right: 16x2 array of right joystick positions (-1 to 1)
    - buttons: 16x17 array of button states (boolean)
    - token_set: List of button names
    
    Returns:
    - Visualization as numpy array
    """
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Create visualization area
    viz_width = min(500, frame_width)
    combined_width = frame_width + viz_width
    combined_height = frame_height
    
    # Create combined image (frame + visualization)
    combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Put the frame on the left side
    combined[:frame_height, :frame_width] = frame
    
    # Starting position for visualizations
    viz_x = frame_width
    viz_y = 20
    
    # Draw joysticks if data is provided
    if i < len(j_left) and i < len(j_right):
        # Add section title
        cv2.putText(combined, "JOYSTICKS", 
                   (viz_x + 10, viz_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        viz_y += 30  # Move down after title
        
        # Size of joystick visualization
        joy_size = min(120, viz_width // 3)
        
        # Horizontal positions of joysticks
        joy_left_x = viz_x + 30
        joy_right_x = viz_x + viz_width - joy_size - 30
        
        # Draw joystick labels
        cv2.putText(combined, "Left", (joy_left_x, viz_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(combined, "Right", (joy_right_x, viz_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Draw joysticks
        draw_joystick(combined, joy_left_x, viz_y, joy_size, j_left[i])
        draw_joystick(combined, joy_right_x, viz_y, joy_size, j_right[i])
        
        viz_y += joy_size + 40  # Move down after joysticks
    
    # Draw buttons if data is provided
    if buttons is not None and i < len(buttons):
        # Add section title
        cv2.putText(combined, "BUTTON STATES", 
                   (viz_x + 10, viz_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        viz_y += 30  # Move down after title
        
        # Size and position of button grid
        button_grid_x = viz_x + 20
        button_grid_y = viz_y
        button_size = 20
        
        # Draw button grid
        draw_button_grid(combined, button_grid_x, button_grid_y, 
                        button_size, buttons, i, token_set)
    
    return combined

def draw_joystick(img, x, y, size, position):
    """Draw a joystick visualization at the specified position."""
    # Draw joystick background
    cv2.rectangle(img, (x, y), (x + size, y + size), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + size, y + size), (100, 100, 100), 1)
    
    # Calculate center point
    mid_x = x + size // 2
    mid_y = y + size // 2
    
    # Draw center cross (0,0 coordinates)
    cv2.line(img, (x, mid_y), (x + size, mid_y), (150, 150, 150), 1)
    cv2.line(img, (mid_x, y), (mid_x, y + size), (150, 150, 150), 1)
    
    # Draw 2x2 grid
    quarter_x = x + size // 4
    quarter_y = y + size // 4
    three_quarters_x = x + 3 * size // 4
    three_quarters_y = y + 3 * size // 4
    
    # Draw grid lines
    cv2.line(img, (quarter_x, y), (quarter_x, y + size), (100, 100, 100), 1)
    cv2.line(img, (three_quarters_x, y), (three_quarters_x, y + size), (100, 100, 100), 1)
    cv2.line(img, (x, quarter_y), (x + size, quarter_y), (100, 100, 100), 1)
    cv2.line(img, (x, three_quarters_y), (x + size, three_quarters_y), (100, 100, 100), 1)
    
    # Draw joystick position (clamp coordinates to valid range)
    px = max(-1, min(1, position[0]))
    py = max(-1, min(1, position[1]))
    
    joy_x = int(mid_x + px * size // 2)
    joy_y = int(mid_y - py * size // 2)  # Y is inverted in image coordinates
    
    # Draw joystick position as a dot
    cv2.circle(img, (joy_x, joy_y), 5, (0, 0, 255), -1)  # Red dot

def draw_button_grid(img, x, y, button_size, buttons, current_row, token_set):
    """Draw the button state grid."""
    rows, cols = buttons.shape
    
    # Ensure the grid fits in the visualization area
    available_width = img.shape[1] - x - 20
    if cols * button_size > available_width:
        button_size = max(10, available_width // cols)
    
    # Draw column numbers at the top
    for col in range(cols):
        number_x = x + col * button_size + button_size // 2
        number_y = y - 5
        cv2.putText(img, str(col + 1), (number_x - 4, number_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Draw button grid
    for row in range(rows):
        for col in range(cols):
            # Calculate button position
            bx = x + col * button_size
            by = y + row * button_size
            
            # Draw button cell
            color = (0, 255, 0) if buttons[row, col] else (0, 0, 0)  # Green if pressed, black otherwise
            cv2.rectangle(img, (bx, by), (bx + button_size, by + button_size), color, -1)
            
            # Draw grid lines
            cv2.rectangle(img, (bx, by), (bx + button_size, by + button_size), (80, 80, 80), 1)
    
    # Highlight current row
    highlight_y = y + current_row * button_size
    cv2.rectangle(img, (x, highlight_y), (x + cols * button_size, highlight_y + button_size), 
                 (0, 0, 255), 2)  # Red highlight
        
    # Draw button legend below the mosaic
    if token_set is not None:
        legend_y = y + rows * button_size + 20  # Starting Y position for legend
        legend_x = x  # Starting X position for legend
        line_height = 15  # Height of each legend line
        
        # Add legend title
        cv2.putText(img, "Button Legend:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        legend_y += line_height + 5  # Move down after title
        
        # Calculate how many columns to use for the legend based on available space
        legend_cols = max(1, min(3, cols // 6))  # Use 1-3 columns depending on button count
        legend_items_per_col = (cols + legend_cols - 1) // legend_cols  # Items per column with ceiling division
        
        # Draw legend entries
        for col in range(min(cols, len(token_set))):
            # Calculate position in the legend grid
            legend_col = col // legend_items_per_col
            legend_row = col % legend_items_per_col
            
            # Calculate position
            entry_x = legend_x + legend_col * (available_width // legend_cols)
            entry_y = legend_y + legend_row * line_height
            
            # Add legend entry
            if col < len(token_set):
                cv2.putText(img, f"{col+1}. {token_set[col]}", (entry_x, entry_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

class VideoRecorder:
    def __init__(self, output_file, fps=30, crf=28, preset="fast"):
        """
        Initialize a video recorder using PyAV.
        
        Args:
            output_file (str): Path to save the video file
            fps (int): Frames per second
            crf (int): Constant Rate Factor (0-51, higher means smaller file but lower quality)
            preset (str): Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        """
        self.output_file = output_file
        self.fps = fps
        self.crf = str(crf)
        self.preset = preset
        self.container = av.open(output_file, mode="w")
        self.stream = None
        
    def init_stream(self, width, height):
        """Initialize the video stream with the frame dimensions."""
        self.stream = self.container.add_stream("h264", rate=self.fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "crf": self.crf,
            "preset": self.preset
        }
        
    def add_frame(self, frame):
        """
        Add a frame to the video.
        
        Args:
            frame (numpy.ndarray): Frame as RGB numpy array
        """
        if self.stream is None:
            self.init_stream(frame.shape[1], frame.shape[0])
            
        av_frame = av.VideoFrame.from_ndarray(np.array(frame), format="rgb24")
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
        
    def close(self):
        """Flush remaining packets and close the video file."""
        try:
            if self.stream is not None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        finally:
            self.container.close()

    def __enter__(self):
        """Support for context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the recorder when exiting the context."""
        self.close()
