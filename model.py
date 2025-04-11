
import os
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

import imageio

from grid import HexagonalGrid
from cells import bacteria_step, phagocyte_step
from utils import create_hex_kernel, weight_func, convolve

class Automaton():
    def __init__(self, grid, log_dir):
        self.grid = grid
        self.frames = []
        self.steps_logs = {}
        self.generation = 0
        self.log_dir = log_dir

    def simulate(self, n_interations):  
        self.kernel =  create_hex_kernel(radius=self.grid.grid_radius,
                                        weight_func=weight_func)
        for i in range(n_interations):
            print("Step ", i)
            self.generation = i
            done = self.game_step()
            

            print(f"Num bacterias: {len(self.grid.bacterias)} | Num Phagocytes: {len(self.grid.phagocytes)}")

            total_size_grid = (2*self.grid.grid_radius)**2 #(self.grid._bacteria_grid.shape[0] * self.grid._bacteria_grid.shape[1])

            self.steps_logs[str(i)] = {"num_bact" : len(self.grid.bacterias),
                "num_phag" : len(self.grid.phagocytes),
                "proportion": len(self.grid.bacterias)/(len(self.grid.bacterias)+len(self.grid.phagocytes)),
                "density_phag": len(self.grid.phagocytes)/total_size_grid,
                "density_bact": len(self.grid.bacterias)/total_size_grid}
            
            
            if i%25 == 0:
                print("Saving...")
                self.get_animation(duration = 50)
                self.save_results()

            if done:
                break
        print("End of simulation")


    def game_step(self):
        if len(self.grid.bacterias) == 0 or (len(self.grid.phagocytes) == 0 and self.generation >150): #or len(self.grid.bacterias.keys())/(2*self.grid.grid_radius)**2>0.7:
            return True
        self.grid.defend(self.generation)
        
        self.grid._get_dense_grids()
        self.grid.set_grads(self.kernel)
        
        phagocytes = {}
        bacterias = {}
        for i, (pos,cell) in enumerate(self.grid.bacterias.items()):
            new_pos, life, new_bact_pos = bacteria_step(self.grid,
                                                        pos,
                                                        cell["life"],
                                                        bacterias,
                                                        phagocytes)
            if new_pos is not None:
                if new_pos not in bacterias:
                    cell["life"] = life
                    bacterias[new_pos] = cell
                else:
                    cell["life"] = life
                    bacterias[pos] = cell
                
            if new_bact_pos is not None:
                bacterias[new_bact_pos] = {"life" : self.grid.bacteria_lifetime}
        for pos,cell in self.grid.phagocytes.items():
            new_pos, life = phagocyte_step(self.grid,
                                           pos,
                                           cell["life"],
                                           bacterias,
                                           phagocytes)
            if new_pos is not None:
                if new_pos not in phagocytes:
                    cell["life"] = life
                    phagocytes[new_pos] = cell
                else:
                    cell["life"] = life
                    phagocytes[pos] = cell

        self.grid.update_config(phagocytes, bacterias)
        
        img = self.grid.plot_grid()
        self.frames.append(img)

        return False



    def get_animation(self, duration=500):
        # fig, ax = plt.subplots()
        # img = ax.imshow(self.frames[0], animated=True)

        # def update(frame):
        #     img.set_array(frame)
        #     return img,

        # ani = animation.FuncAnimation(fig, update, frames=self.frames, blit=True, interval=duration)
        
        # # Save as MP4 using the "pillow" writer
        # ani.save(os.path.join(self.log_dir, "animation.mp4"), writer="pillow", fps=30)

        imageio.mimsave(os.path.join(self.log_dir, "animation.mp4"), self.frames, fps=40)

    def save_results(self):
        
        proportion = []
        densities_bact = []
        densities_phag = []
        b_list = []
        p_list = []
        for k, infos in self.steps_logs.items():
            b = infos["num_bact"]
            p = infos["num_phag"]
            b_list.append(b)
            p_list.append(p)
            proportion.append(infos["proportion"])
            densities_phag.append(infos["density_phag"])
            densities_bact.append(infos["density_bact"])
        
        fig = plt.figure(figsize=(10,10))
        plt.plot([i for i in self.steps_logs.keys()],
                 proportion)
        plt.xticks(range(0, len(self.steps_logs),25))
        
        fig.savefig(os.path.join(self.log_dir,"proportions.png"))
        
        fig = plt.figure(figsize=(10,10))
        plt.plot([i for i in self.steps_logs.keys()],
                 b_list, label = "Bacterias")
        
        plt.plot([i for i in self.steps_logs.keys()],
                 p_list, label = "Phagocytes")

        plt.xticks(range(0, len(self.steps_logs),25))
        
        plt.legend()
        fig.savefig(os.path.join(self.log_dir,"numbers.png"))

        fig = plt.figure(figsize=(10,10))
        plt.plot([i for i in self.steps_logs.keys()],
                 densities_bact, label = "Bacterias")
        
        plt.plot([i for i in self.steps_logs.keys()],
                 densities_phag, label = "Phagocytes")

        plt.xticks(range(0, len(self.steps_logs),25))
        
        plt.legend()
        
        fig.savefig(os.path.join(self.log_dir,"densities.png"))
        with open(os.path.join(self.log_dir, "results.json"), "w+") as f:
            json.dump(self.steps_logs, f)




@hydra.main(config_path="/home/damemilien/Documents/emilien", config_name="config", version_base=None)
def main(cfg: DictConfig):

    seed = cfg.seed 
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    now = datetime.now()
    log_dir = cfg.logging.base_dir
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "params.json"), "w+") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f)
    
    env_grid = HexagonalGrid(
        device=device,
        grid_radius=cfg.params.grid_radius,
        bacteria_lifetime=cfg.params.bacteria_lifetime,
        phagocytes_life_ratio=cfg.params.life_ratio,
        max_num_phagocytes=cfg.params.max_num_phagocytes,
        phagocytes_birth_distance=cfg.params.phagocytes_birth_distance,
        phagocytes_rate_type=cfg.params.phag_rate_type,
        phagocytes_spwan_rate=cfg.params.phagocytes_spwan_rate,
        phagocytes_starting_number=cfg.params.phagocytes_starting_number
    )
    
    env_grid.infect(
        num_bacterias=cfg.params.initial_bacterias,
        infection_radius=int(np.sqrt(cfg.params.initial_bacterias / cfg.params.initial_dentisity))
    )
    
    automaton = Automaton(env_grid, log_dir)
    automaton.simulate(cfg.simulation.steps)
    automaton.get_animation(duration=cfg.simulation.animation_duration)
    automaton.save_results()

if __name__ == "__main__":
    main()


