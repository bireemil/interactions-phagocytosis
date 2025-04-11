from PIL import Image
import io
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import math

import torch

from utils import linear_gradient, convert_string_to_tuple, lerp, create_hex_kernel, weight_func, convolve

class HexagonalGrid():
    # Axial coordinate system (q, r)
    DIRECTIONS = [
        (1, 0), (1, -1), (0, -1), 
        (-1, 0), (-1, 1), (0, 1)
    ]

    def __init__(self,
                 device,
                 bacteria_lifetime = 10,
                 grid_radius = 100, 
                 phagocytes_life_ratio = 1,
                 phagocytes_birth_distance = 2,
                 phagocytes_rate_type = "linear",
                 phagocytes_starting_number = 10,
                 max_num_phagocytes = 500,
                 phagocytes_spwan_rate = 1):
        
        self.device = device
        self.grid_radius = grid_radius
        self.contour = self._get_contour()
        self.bacterias = {}
        self.phagocytes = {}
        self.radius_of_infection = 0
        self.barycenter_of_infection = (0,0)
        self.bacteria_lifetime = bacteria_lifetime
        self.infection_radius = []
        self.infection_centers = []

        self.phagocytes_life_ratio = phagocytes_life_ratio 
        self.phagocytes_starting_number = phagocytes_starting_number
        self.phagocytes_rate_type = phagocytes_rate_type
        self.phagocytes_birth_distance = phagocytes_birth_distance
        self.max_num_phagocytes = max_num_phagocytes
        self.phagocytes_spwan_rate = phagocytes_spwan_rate
        self.phag_center = None
        
        
        self.current_center_figure = (0,0)
        self.current_lim_figure = 0

    def _get_contour(self):
        """Returns the contour positions of a hexagon at a given radius."""
        if self.grid_radius < 1:
            return [str((0, 0))]  # A self.grid_radius of 0 means only the center cell exists
        
        # Hexagonal directions in axial coordinates
        
        # Start at (self.grid_radius, 0)
        contour = []
        q, r = -self.grid_radius, self.grid_radius
        
        # Traverse the hexagon edges
        for dq, dr in self.DIRECTIONS:
            for _ in range(self.grid_radius):
                contour.append(str((q, r)))
                q += dq
                r += dr
        
        return contour


    def _get_dense_grids(self):
        
        q_min_b, q_max_b = 0,0
        r_min_b, r_max_b= 0,0
        
        q_min_p, q_max_p = 0,0
        r_min_p, r_max_p = 0,0
        
        bact_tuple = []
        phag_tuple = []
        
        if len(self.bacterias.keys()) >0 : 
            bact_tuple = [convert_string_to_tuple(k) for k in self.bacterias.keys()] 
            qs, rs = zip(*bact_tuple)
            q_min_b, q_max_b = min(qs), max(qs)
            r_min_b, r_max_b = min(rs), max(rs)
        
        if len(self.phagocytes.keys()) >0:
            phag_tuple = [convert_string_to_tuple(k) for k in self.phagocytes.keys()]
            qs, rs = zip(*phag_tuple)
            q_min_p, q_max_p = min(qs), max(qs)
            r_min_p, r_max_p = min(rs), max(rs)
        
        self.q_min = -self.grid_radius -1 #min(q_min_p, q_min_b) - 1
        self.q_max = self.grid_radius +1 #max(q_max_p, q_max_b) + 1
        self.r_min = -self.grid_radius -1 #min(r_min_p, r_min_b) - 1
        self.r_max = self.grid_radius +1 #max(r_max_p, r_max_b) + 1
        
        self.q_size = self.q_max - self.q_min + 1
        self.r_size = self.r_max - self.r_min + 1
        self._bacteria_grid = torch.zeros((self.q_size, self.r_size), device=self.device)
        self._phagocytes_grid = torch.zeros((self.q_size, self.r_size), device=self.device)
        
        # Fill dense tensor
        for (q, r) in bact_tuple:
            q_idx = q - self.q_min
            r_idx = r - self.r_min
            self._bacteria_grid[q_idx, r_idx] = 1
        
        for (q, r) in phag_tuple:
            q_idx = q - self.q_min 
            r_idx = r - self.r_min
            self._phagocytes_grid[q_idx, r_idx] = 1
            
    def set_grads(self, kernel = None):
        
        if kernel is None:
            self.kernel =  create_hex_kernel(radius=max(self.q_size, self.r_size)//2,
                                                weight_func=weight_func)
            
            kernel = self.kernel
        
        
        self.bact_grads = convolve(self._bacteria_grid,
                                    kernel)
        
        self.phag_grads = convolve(self._phagocytes_grid,
                                    kernel)
            
                    
    
    def add_cell(self, pos, cell_type, cell_dict):
        if cell_type == 0:
            self.bacterias[pos] = cell_dict
        
        elif cell_type == 1:
            self.phagocytes[pos] = cell_dict
    
    def infect(self,
               num_bacterias,
               infection_radius,
               center = (0,0)):
        
        self.infection_centers.append(center)
        self.infection_radius.append(infection_radius)

        possible_positions = []
        for q in range(center[0]-infection_radius, center[1] + infection_radius + 1):
            r1 = max(center[0]-infection_radius, -q - center[1] - infection_radius)
            r2 = min(center[0] + infection_radius, -q + center[1] + infection_radius)
            for r in range(r1, r2 + 1):
                possible_positions.append(str((q,r)))

        initial_pos_idx = np.random.choice(range(len(possible_positions)), int(num_bacterias), replace=False)
        initial_bact_pos = np.array(possible_positions)[initial_pos_idx]

        for pos in initial_bact_pos:
            bact = {"life": self.bacteria_lifetime}
            
            self.add_cell(pos, 0, bact)

        bact_pos = np.array([convert_string_to_tuple(pos) for pos in self.bacterias])
        radius_of_infection = np.max(np.abs(np.std(bact_pos, axis=0)))
        self.radius_of_infection = radius_of_infection
        
        self.current_lim_figure = radius_of_infection
    
    def defend(self,
               generation):
        if self.phagocytes_rate_type == "linear":
            if len(self.bacterias)/(len(self.phagocytes) +1)>self.phagocytes_starting_number:
                if self.max_num_phagocytes>=1:
                    num_new_phagocytes = int(self.max_num_phagocytes)
                else:
                    p = np.random.random()
                    if p > 1-self.max_num_phagocytes:
                        num_new_phagocytes = 1
                    else:
                        num_new_phagocytes = 0
            else:
                num_new_phagocytes = 0
            
        if self.phagocytes_rate_type == "sigmoid":
            # target_num = int(self.max_num_phagocytes/(1+np.exp(-self.phagocytes_spwan_rate * (generation) + self.bacteria_lifetime//2)))
            numerator = self.max_num_phagocytes * (1 - np.exp(-self.phagocytes_spwan_rate * len(self.bacterias)))
            denominator = (0.5 +  np.exp(-0.005 * generation))
            num_new_phagocytes =  int(numerator/denominator) 

        if self.phagocytes_rate_type == "spikes":
            num_new_phagocytes = self.phagocytes_starting_number * (generation%(self.bacteria_lifetime//2) == 0)

        bact_pos = np.array([convert_string_to_tuple(pos) for pos in self.bacterias])
        barycenter_of_infection = np.mean(bact_pos, axis=0)
        barycenter_of_infection = (int(barycenter_of_infection[0]), int(barycenter_of_infection[1]))
        radius_of_infection = np.max(np.abs(np.std(bact_pos, axis=0)))
        self.radius_of_infection = radius_of_infection
        self.barycenter_of_infection = barycenter_of_infection

        if num_new_phagocytes>0: 
            spawn_radius = int(np.sqrt(self.phagocytes_starting_number))
            phagocyte_positions = [pos for pos in self.contour if pos not in self.bacterias
                                   and pos not in self.phagocytes]
            phagocyte_positions = np.random.choice(phagocyte_positions,
                                                  min(num_new_phagocytes, len(phagocyte_positions)),
                                                 replace=False)


            # phag_center = self.random_point_at_distance(str(barycenter_of_infection),
            #                                             self.grid_radius-1)
            # phag_center = convert_string_to_tuple(phag_center)
            # self.phag_center = phag_center

            # for q in range(self.phag_center[0], self.phag_center[0]):
            #     for r in range(self.phag_center[1], self.phag_center[1]):
            #         if str((q,r)) not in self.phagocytes:
                        # phagocyte_positions.append(str((q, r)))
            

            # phagocyte_pos_idx = np.random.choice(len(phagocyte_positions),
            #                                     min(num_new_phagocytes, len(phagocyte_positions)),
            #                                     replace=False)
            
            # phagocyte_positions = np.array(phagocyte_positions)[phagocyte_pos_idx]
            
            # Populate the grid
            for pos in phagocyte_positions:
                phag = {"life": self.bacteria_lifetime * self.phagocytes_life_ratio}
                
                self.add_cell(pos, 1, phag)
    


    def get_populated_cells(self):  
        return [k for k in self.phagocytes.keys()] + [k for k in self.bacterias.keys()]

    def update_config(self, phagocytes, bacterias):
        # self.cells = self.cells.clear()
        self.bacterias = bacterias
        self.phagocytes = phagocytes
        
        
    def get_phag_grad(self,pos):
        
        neighbors = self.get_neighbors(pos)
        phag_grad = {k:0 for k in neighbors
                     if k not in self.bacterias and
                     k not in self.phagocytes}

        # phag_sample = np.random.choice(a=list(self.phagocytes.keys()), size = min(100, len(self.phagocytes)))
        
        # for k in phag_grad:
        #     for phag_pos in phag_sample:
        #         phag_grad[k] += linear_gradient(convert_string_to_tuple(phag_pos),
        #                                         convert_string_to_tuple(k))
                
        #     phag_grad[k] = phag_grad[k]
        
        for pos in phag_grad:
            q,r = convert_string_to_tuple(pos)
            phag_grad[pos] = self.phag_grads[q-self.q_min,r-self.r_min] 
        
        return phag_grad
    
    def get_bact_grad(self,pos):
        neighbors = self.get_neighbors(pos)
        bact_grad = {k:0 for k in neighbors
                     if k not in self.bacterias and
                     k not in self.phagocytes}
        
        # bact_sample = np.random.choice(a=list(self.bacterias.keys()), size = min(250, len(self.bacterias)))

        # for bact_pos in bact_sample:
        #     for k in bact_grad:
        #         bact_grad[k] += linear_gradient(convert_string_to_tuple(bact_pos),
        #                                         convert_string_to_tuple(k))
        for pos in bact_grad:
            q,r = convert_string_to_tuple(pos)
            # print(q, self.q_min, q-self.q_min, self.bact_grads.shape)
            # print(r, self.r_min, r-self.r_min)
            bact_grad[pos] = self.bact_grads[q - self.q_min,r - self.r_min] 
        return bact_grad

    def __len__(self):
        return len(self.cells.keys())

    def get_neighbors(self, pos, distance=1):
        """Get neighboring tile coordinates within a bounded hexagonal grid."""
        pos = convert_string_to_tuple(pos)
        q, r = pos

        neighbors = [
            (q + int(distance) * dq, r + int(distance) * dr)
            for dq, dr in self.DIRECTIONS
        ]

        if self.grid_radius is not None:
            neighbors = [
                str((qn, rn)) for qn, rn in neighbors 
                if (abs(qn) + abs(rn) + abs(-qn - rn))//2 <= self.grid_radius
            ]
        else:
            neighbors = [str(n) for n in neighbors]

        return neighbors

    def random_point_at_distance(self, pos, distance):
        q0, r0 = convert_string_to_tuple(pos)
        if distance == 0:
            return str((q0, r0))

        # Start at (q0, r0) and move distance steps in one direction
        q, r = q0 + distance * self.DIRECTIONS[4][0], r0 + distance * self.DIRECTIONS[4][1]
        ring = []

        # Walk around the hex ring
        for d in range(6):  # Six self.directions
            for _ in range(distance):
                ring.append(str((q, r)))
                q += self.DIRECTIONS[d][0]
                r += self.DIRECTIONS[d][1]
        return np.random.choice(ring) if ring else None

    def distance(self, q1, r1, q2, r2):
        return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2
    
    def random_hex_at_distance(self, pos, distance):
        """
        Select a random hex tile that is exactly at a given distance from (q0, r0).
        
        :param q0: Axial coordinate q of the center tile
        :param r0: Axial coordinate r of the center tile
        :param distance: The exact distance from (q0, r0)
        :return: A random (q, r) coordinate
        """
        candidates = self.get_neighbors(pos, distance)
        return np.random.choice(candidates) if candidates else None
    
    def plot_grid(self):
        """Visualize the hexagonal grid more efficiently"""
        fig, ax = plt.subplots(figsize=(15, 15))
        # ax.set_title(f"Hexagonal Grid")
        
        hex_height = math.sqrt(3)/2
        hex_width = 1

        patches_list = []
        colors = []
        
        # bact_grad_dict = {}
        # phag_grad_dict = {}
        
        # # print(self.bact_grads)
        
        # for q in range(self.q_size):
        #     for r in range(self.r_size):
        #         if self.bact_grads[q, r] != 0:
        #             bact_grad_dict[str((q + self.q_min, r + self.r_min))] = self.bact_grads[q, r].item()/self.bact_grads.max().item()
        #         if self.phag_grads[q, r] != 0:
        #             phag_grad_dict[str((q + self.q_min, r + self.r_min))] = self.phag_grads[q, r].item()/self.phag_grads.max().item()
                    
        cmap_phag = cm.get_cmap("inferno")
        cmap_bact = cm.get_cmap("gist_heat")
        
        # Precompute hexagon shape
        angle_deg = 60
        base_hex = [(math.cos(math.radians(angle_deg * i)), math.sin(math.radians(angle_deg * i))) for i in range(6)]
        
        # for pos in bact_grad_dict:
        #     q,r = convert_string_to_tuple(pos)
        #     x = hex_width * (3/2 * q)
        #     y = hex_height * (2 * r + q)
        #     value = bact_grad_dict[pos]
        #     colors.append(cmap_bact(value))
        #     hex_points = [(px + x, py + y) for px, py in base_hex]
        #     polygon = patches.Polygon(hex_points, closed=True, edgecolor=(0,0,0,0.1))
        #     patches_list.append(polygon)
        
        max_grad_phag = self.phag_grads.cpu().max()
        max_grad_bact = self.bact_grads.cpu().max()
        # for q in range(self.grid_radius):
        #     for r in range(self.grid_radius):
        #         # if self.distance(self.grid_radius//2, self.grid_radius//2, q,r) <= self.grid_radius//2: 
        #             x = hex_width * (3/2 * (q + self.q_min))
        #             y = hex_height * (2 * (r+self.r_min) + (q + self.q_min))
        #             ax.text(x, y, f"{x:.1f},{y:.1f}", fontsize=15, ha="center", va="center", color="black", weight="bold")

        for q in range(self.bact_grads.shape[0]):
            for r in range(self.bact_grads.shape[1]):
            # for pos in self.get_populated_cells():
                # q,r = convert_string_to_tuple(pos)
                if self.distance(self.grid_radius,self.grid_radius, q,r) <= self.grid_radius:
                    x = hex_width * (3/2 * (q + self.q_min))
                    y = hex_height * (2 * (r+self.r_min) + (q + self.q_min))
                    # Translate base hexagon to correct position
                    hex_points = [(px + x, py + y) for px, py in base_hex]
                    polygon = patches.Polygon(hex_points, closed=True, edgecolor=(0,0,0,0))
                    patches_list.append(polygon)
                    # print(self._bacteria_grid[q,r].cpu()/max_grad)
                    # colors.append(cmap_bact(self.bact_grads[q,r].cpu()/max_grad))

                    grad_value_phag, grad_value_bact = self.phag_grads[q,r].cpu()/max_grad_phag, self.bact_grads[q,r].cpu()/max_grad_bact
                    if grad_value_phag>grad_value_bact:
                        colors.append(cmap_phag(grad_value_phag))
                    else:
                        colors.append(cmap_bact(grad_value_bact))
                # # Determine color
                # key = str((q, r))
                # if key in self.bacterias:
                #     colors.append("red")
                # elif key in self.phagocytes:
                #     colors.append("green")
            # else:
            #     value = sum(linear_gradient(convert_string_to_tuple(b), (q, r)) for b in self.bacterias) / max(1, len(self.bacterias))
            #     colors.append("white")

        # Add all patches at once
        patch_collection = PatchCollection(patches_list, facecolor=colors, match_original=True)
        ax.add_collection(patch_collection)

        self.alpha = 0.03
        # self.current_center_figure = tuple(lerp(np.array(self.current_center_figure), np.array(self.barycenter_of_infection), self.alpha))
        # self.current_lim_figure = lerp(self.current_lim_figure, self.radius_of_infection, self.alpha)
        self.current_lim_figure = self.grid_radius
        ax.set_xlim(-self.current_lim_figure * 2 + self.current_center_figure[0],
                    self.current_lim_figure * 2 + self.current_center_figure[0])
        
        ax.set_ylim(-self.current_lim_figure * 2 + self.current_center_figure[1],
                    self.current_lim_figure * 2 + + self.current_center_figure[1])
        
        ax.set_aspect('equal')
        ax.set_facecolor("black")
        
        ax.grid(False)
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_yticklabels([])  # Remove y-axis labels
        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['left'].set_visible(False)  # Hide the left spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        fig.tight_layout()
        # plt.axis("off")
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Open the image from the buffer and append it to the frames list
        img = Image.open(buf)
        plt.close(fig)
        return img

        # plt.show()

def create_hexagonal_map(radius):
    """Create a hexagonal map with a given radius"""
    grid = HexagonalGrid()
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            grid.add_tile(q, r)
    return grid

# Example usage
if __name__ == "__main__":
    # Create a hexagonal grid with radius 3
    hex_grid = create_hexagonal_map(3)
    print(len(hex_grid))
    # Get neighbors of a specific tile
    test_tile = (1, 1)
    neighbors = hex_grid.get_neighbors(*test_tile)
    for n in neighbors:
        print(linear_gradient(test_tile, n))
    print(f"Neighbors of tile {test_tile}: {neighbors}")

    hex_grid.plot_grid(10)