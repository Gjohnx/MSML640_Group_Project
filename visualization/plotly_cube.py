import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple

# Rubik's Cube face colors
FACE_COLORS = {
    "U": "white",
    "D": "yellow",
    "F": "green",
    "B": "blue",
    "L": "orange",
    "R": "red",
}

class PlotlyCube:
    
    def __init__(self, size=3):
        self.size = size
        self.cubelets = []
        self.build_cube()
        
    def build_cube(self):
        self.cubelets = []
        offset = (self.size - 1) / 2
        
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    pos = np.array([x - offset, y - offset, z - offset])
                    self.cubelets.append({
                        'position': pos,
                        'colors': self._get_initial_colors(pos)
                    })
    
    def _get_initial_colors(self, pos):
        colors = {}
        cubelet_size = 0.80
        
        # U (top, +y)
        if pos[1] > 0.99:
            colors['U'] = FACE_COLORS['U']
        # D (bottom, -y)
        if pos[1] < -0.99:
            colors['D'] = FACE_COLORS['D']
        # F (front, +z)
        if pos[2] > 0.99:
            colors['F'] = FACE_COLORS['F']
        # B (back, -z)
        if pos[2] < -0.99:
            colors['B'] = FACE_COLORS['B']
        # L (left, -x)
        if pos[0] < -0.99:
            colors['L'] = FACE_COLORS['L']
        # R (right, +x)
        if pos[0] > 0.99:
            colors['R'] = FACE_COLORS['R']
            
        return colors
    
    def _create_cubelet_mesh(self, position, colors, size=0.80):
        meshes = []
        gap = 0.01
        s = size
        
        # Create the black body/edges
        vertices, faces = self._cube_geometry(position, s - gap)
        meshes.append(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='black',
            opacity=1,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add colored stickers on visible faces
        sticker_size = s * 0.99
        sticker_thickness = 0.02
        
        for face, color in colors.items():
            if face == 'U':  # Top (+y)
                sticker_pos = position + np.array([0, s, 0])
                v, f = self._flat_square_geometry(sticker_pos, sticker_size, sticker_thickness, 'y')
            elif face == 'D':  # Bottom (-y)
                sticker_pos = position + np.array([0, -s, 0])
                v, f = self._flat_square_geometry(sticker_pos, sticker_size, sticker_thickness, 'y')
            elif face == 'F':  # Front (+z)
                sticker_pos = position + np.array([0, 0, s])
                v, f = self._flat_square_geometry(sticker_pos, sticker_size, sticker_thickness, 'z')
            elif face == 'B':  # Back (-z)
                sticker_pos = position + np.array([0, 0, -s])
                v, f = self._flat_square_geometry(sticker_pos, sticker_size, sticker_thickness, 'z')
            elif face == 'L':  # Left (-x)
                sticker_pos = position + np.array([-s, 0, 0])
                v, f = self._flat_square_geometry(sticker_pos, sticker_size, sticker_thickness, 'x')
            elif face == 'R':  # Right (+x)
                sticker_pos = position + np.array([s, 0, 0])
                v, f = self._flat_square_geometry(sticker_pos, sticker_size, sticker_thickness, 'x')
            
            meshes.append(go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=color,
                opacity=1,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        return meshes
    
    def _cube_geometry(self, center, size):
        """Generate vertices and faces for a cube."""
        s = size / 2
        vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # back
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]   # front
        ]) + center
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # back
            [4, 5, 6], [4, 6, 7],  # front
            [0, 1, 5], [0, 5, 4],  # bottom
            [2, 3, 7], [2, 7, 6],  # top
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ])
        
        return vertices, faces
    
    def _flat_square_geometry(self, center, size, thickness, normal_axis):
        s = size / 2
        t = thickness / 1
        
        if normal_axis == 'y':
            vertices = np.array([
                [-s, -t, -s], [s, -t, -s], [s, -t, s], [-s, -t, s],
                [-s, t, -s], [s, t, -s], [s, t, s], [-s, t, s]
            ]) + center
        elif normal_axis == 'z':
            vertices = np.array([
                [-s, -s, -t], [s, -s, -t], [s, s, -t], [-s, s, -t],
                [-s, -s, t], [s, -s, t], [s, s, t], [-s, s, t]
            ]) + center
        else:  # x
            vertices = np.array([
                [-t, -s, -s], [-t, s, -s], [-t, s, s], [-t, -s, s],
                [t, -s, -s], [t, s, -s], [t, s, s], [t, -s, s]
            ]) + center
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # front
            [4, 5, 6], [4, 6, 7],  # back
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5]
        ])
        
        return vertices, faces
    
    def create_figure(self, title="Rubik's Cube Visualization"):
        """Create a Plotly figure with the current cube state."""
        meshes = []
        
        for cubelet in self.cubelets:
            cubelet_meshes = self._create_cubelet_mesh(
                cubelet['position'],
                cubelet['colors']
            )
            meshes.extend(cubelet_meshes)
        
        fig = go.Figure(data=meshes)
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=[-2, 2]),
                yaxis=dict(visible=False, range=[-2, 2]),
                zaxis=dict(visible=False, range=[-2, 2]),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            width=300,
            showlegend=False
        )
        
        return fig

def create_cube_visualization() -> go.Figure:
    cube = PlotlyCube(size=3)
    
    return cube.create_figure()

