import json
import numpy as np
from pathlib import Path
import yaml


def make_base_dict(
    image_name: str,
    width: int,
    height: int,
    tag: int = 0,
    depth: int = 3,
):
    return {
        "image": image_name,
        "tag": tag,
        "size": {
            "depth": depth,
            "height": height,
            "width": width,
        },
    }


def add_object(
    yaml_dict: dict,
    points: np.ndarray,
    width: int,
    height: int,
    name: str,
):
    assert points.ndim == 2
    assert points.shape[1] == 2

    if "objects" not in yaml_dict:
        yaml_dict["objects"] = []

    xy = points / [width, height]
    xy = np.round(np.clip(xy, 0, 1), 6)
    yaml_dict["objects"].append(
        {
            "bpol": {
                "px": xy[:, 0].tolist(),
                "py": xy[:, 1].tolist(),
            },
            "name": name,
        }
    )


def add_text_object(
    yaml_dict: dict,
    points: np.ndarray,
    width: int,
    height: int,
    name: str,
    text: str,
):
    add_object(
        yaml_dict,
        points,
        width,
        height,
        name,
    )

    yaml_dict["objects"][-1]["text"] = text


def open_yaml(yaml_path: Path) -> dict:
    try:
        with open(yaml_path, "r") as yaml_file:
            yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        raise e
    return yaml_dict


def save_yaml(yaml_dict: dict, yaml_path: Path, default_flow_style=None):
    try:
        with open(yaml_path, mode="wt", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_flow_style=default_flow_style,
            )

    except yaml.YAMLError as e:
        raise e


#def polygon_list_array(points: list):
#    assert len(points) % 2 == 0
#    return np.array(points).reshape((-1, 2))


#def min_size_polygon(xy: np.ndarray):
#    return np.min(np.linalg.norm(xy - np.roll(xy, -1, axis=0), axis=1))


def pretty_print_dict(d: dict, sort_keys: bool = True, indent: int = 2):
    print(json.dumps(d, sort_keys=sort_keys, indent=indent))
