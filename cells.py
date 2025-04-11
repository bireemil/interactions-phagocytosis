import numpy as np
import torch

def bacteria_step(grid, pos, life, temp_pos_bact, temp_pos_phag):
    if life <= 0:
        return None, None, None
    
    neighbors = grid.get_neighbors(pos)
    phagocytes_neighbors = [k for k in neighbors
                        if k in grid.phagocytes]
    if len(phagocytes_neighbors) > 0:
        return (pos, life, None)

    vacant_neighboors = [k for k in neighbors
                        if (k not in grid.bacterias
                            and k not in grid.phagocytes
                            and k not in temp_pos_bact
                            and k not in temp_pos_phag)]
    
    if len(vacant_neighboors) > 0:
        prob_to_replicate = len(vacant_neighboors)/(len(neighbors)+1)
        if np.random.random() < prob_to_replicate:
            repl_idx = np.random.randint(len(vacant_neighboors))
            new_bact_pos = vacant_neighboors[repl_idx]
            life = grid.bacteria_lifetime
            return (pos, life, new_bact_pos)
        
        else:
            phagocytes_grad = grid.get_phag_grad(pos)
            combined_grad = {k:1/vp for k,vp in phagocytes_grad.items()}
            # combined_grad = {k:vb for k,vb in bacteria_grad.items()}
            
            best_positions = sorted(combined_grad.keys(),
                                    key = lambda k: combined_grad[k],
                                    reverse = True)
            for next_pos in best_positions:
                if not next_pos in grid.bacterias \
                and not next_pos in grid.phagocytes \
                and not next_pos in temp_pos_phag \
                and next_pos not in temp_pos_bact \
                and combined_grad[next_pos] != torch.inf:
                    # pos = next_pos
                    return next_pos, life, None
    
    return (pos, life, None)


def phagocyte_step(grid, pos, life, temp_pos_bact, temp_pos_phag):
    if life <=0:
        return None, None

    neighbors = grid.get_neighbors(pos)
    bacteria_neighbors = np.array([k for k in neighbors
                        if k in grid.bacterias])
    
    if len(bacteria_neighbors) > 0:
        if len(bacteria_neighbors)/len(neighbors)<5/6:
        
            target_bact_idx = np.random.choice(range(len(bacteria_neighbors)),
                                                size=min(len(bacteria_neighbors),6),
                                                replace=False)
            target_bact_pos = bacteria_neighbors[target_bact_idx]
            for bact_pos in target_bact_pos:
                target_bact = grid.bacterias[bact_pos]
                target_bact["life"] -= 1
                # life -= 1/len(bacteria_neighbors)
            return pos, life
        
        else:
            return None, None

    # life -= 1
    bacteria_grad = grid.get_bact_grad(pos)
    phagocytes_grad = grid.get_phag_grad(pos)

    combined_grad = {k:vb/vp for k,vb,vp in zip(bacteria_grad.keys(),
                                                bacteria_grad.values(),
                                                phagocytes_grad.values())}
    # combined_grad = {k:vb for k,vb in bacteria_grad.items()}
    
    best_positions = sorted(combined_grad.keys(),
                            key = lambda k: combined_grad[k],
                            reverse = True)
    
    # print(neighbors)
    # print(best_positions)

    for next_pos in best_positions:
        if not next_pos in grid.bacterias \
        and not next_pos in grid.phagocytes \
            and not next_pos in temp_pos_bact \
                and not next_pos in temp_pos_phag:
            # pos = next_pos
            return next_pos, life
        
    else:
        return pos, life