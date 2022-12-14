import os
import sys
import traceback

sys.path.append(".")
sys.path.append("ncs")
from ncs.utils.config import MainConfig
from ncs.global_vars import BODY_DIR, RESULTS_DIR, ROOT_DIR

from blender.utils import clean_scene, load_object, mesh_cache_modifier


def main(config, loc=(0, 0, 0), rot=(0, 0, 0)):
    # Load OBJs
    body_obj = os.path.join(BODY_DIR, config.body.model, "body.obj")
    body_pc2 = os.path.join(RESULTS_DIR, config.name, "body.pc2")
    body = load_object(body_obj, name="body", loc=loc, rot=rot)
    mesh_cache_modifier(body, body_pc2)
    garment_obj = os.path.join(
        BODY_DIR, config.body.model, config.garment.name + ".obj"
    )
    garment_pc2 = os.path.join(RESULTS_DIR, config.name, config.garment.name + ".pc2")
    garment_unskinned_pc2 = os.path.join(
        RESULTS_DIR, config.name, config.garment.name + "_unskinned.pc2"
    )
    garment = load_object(garment_obj, name=config.name, loc=loc, rot=rot)
    mesh_cache_modifier(garment, garment_pc2)
    garment.active_material.diffuse_color = (0.5, 0.5, 1.0, 1.0)

    loc = (loc[0] + 1, *loc[1:])
    garment_unskinned = load_object(
        garment_obj, name=config.name + "_unskinned", loc=loc, rot=rot
    )
    mesh_cache_modifier(garment_unskinned, garment_unskinned_pc2)
    garment_unskinned.active_material.diffuse_color = (0.5, 0.5, 1.0, 1.0)


if __name__ == "__main__":
    clean_scene()
    results_folder = os.path.join(ROOT_DIR, "results")
    configs = [arg for arg in sys.argv if arg.endswith(".json")]
    for i, config in enumerate(configs):
        try:
            config = MainConfig(config)
            main(config, loc=(i, 0, 0), rot=(0, 0, 0))
        except:
            print("Could not load predictions for: ", config)
            print("")
            print(traceback.format_exc())
            print("")
