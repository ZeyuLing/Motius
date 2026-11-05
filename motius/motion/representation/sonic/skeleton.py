# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 skeleton components; see licenses/sonic."""
from abc import abstractmethod
from collections import OrderedDict
import json
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            np.int_
            | np.intc
            | np.intp
            | np.int8
            | np.int16
            | np.int32
            | np.int64
            | np.uint8
            | np.uint16
            | np.uint32
            | np.uint64,
        ):
            return int(obj)
        elif isinstance(obj, np.float_ | np.float16 | np.float32 | np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {
                "__ndarray__": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        return json.JSONEncoder.default(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = np.asarray(dct["__ndarray__"], dtype=dct["dtype"])
        return data.reshape(dct["shape"])
    return dct


class Serializable:
    """Implementation to read/write to file.
    All class the is inherited from this class needs to implement to_dict() and
    from_dict()
    """

    @classmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        """Read the object from an ordered dictionary

        :param dict_repr: the ordered dictionary that is used to construct the object
        :type dict_repr: OrderedDict
        :param args, kwargs: the arguments that need to be passed into from_dict()
        :type args, kwargs: additional arguments
        """
        pass

    @abstractmethod
    def to_dict(self):
        """Construct an ordered dictionary from the object

        :rtype: OrderedDict
        """
        pass

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        """Read the object from a file (either .npy or .json)

        :param path: path of the file
        :type path: string
        :param args, kwargs: the arguments that need to be passed into from_dict()
        :type args, kwargs: additional arguments
        """
        if path.endswith(".json"):
            with open(path) as f:
                d = json.load(f, object_hook=json_numpy_obj_hook)
        elif path.endswith(".npy"):
            d = np.load(path, allow_pickle=True).item()
        else:
            assert False, f"failed to load {cls.__name__} from {path}"
        assert d["__name__"] == cls.__name__, "the file belongs to {}, not {}".format(
            d["__name__"], cls.__name__
        )
        return cls.from_dict(d, *args, **kwargs)

    def to_file(self, path: str) -> None:
        """Write the object to a file (either .npy or .json)

        :param path: path of the file
        :type path: string
        """
        if os.path.dirname(path) != "" and (not os.path.exists(os.path.dirname(path))):
            os.makedirs(os.path.dirname(path))
        d = self.to_dict()
        d["__name__"] = self.__class__.__name__
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(d, f, cls=NumpyEncoder, indent=4)
        elif path.endswith(".npy"):
            np.save(path, d)


class TensorUtils(Serializable):

    @classmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        """Read the object from an ordered dictionary

        :param dict_repr: the ordered dictionary that is used to construct the object
        :type dict_repr: OrderedDict
        :param kwargs: the arguments that need to be passed into from_dict()
        :type kwargs: additional arguments
        """
        return torch.from_numpy(dict_repr["arr"].astype(dict_repr["context"]["dtype"]))

    def to_dict(self):
        """Construct an ordered dictionary from the object

        :rtype: OrderedDict
        """
        return NotImplemented


def tensor_to_dict(x):
    """Construct an ordered dictionary from the object

    :rtype: OrderedDict
    """
    x_np = x.numpy()
    return {"arr": x_np, "context": {"dtype": x_np.dtype.name}}


class SkeletonTree(Serializable):
    """
    A skeleton tree gives a complete description of a rigid skeleton. It describes a tree structure
    over a list of nodes with their names indicated by strings. Each edge in the tree has a local
    translation associated with it which describes the distance between the two nodes that it
    connects.

    Basic Usage:
        >>> t = SkeletonTree.from_mjcf(SkeletonTree.__example_mjcf_path__)
        >>> t
        SkeletonTree(
            node_names=['torso', 'front_left_leg', 'aux_1', 'front_left_foot', 'front_right_leg', 'aux_2', 'front_right_foot', 'left_back_leg', 'aux_3', 'left_back_foot', 'right_back_leg', 'aux_4', 'right_back_foot'],
            parent_indices=tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  0, 10, 11]),
            local_translation=tensor([[ 0.0000,  0.0000,  0.7500],
                    [ 0.0000,  0.0000,  0.0000],
                    [ 0.2000,  0.2000,  0.0000],
                    [ 0.2000,  0.2000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000],
                    [-0.2000,  0.2000,  0.0000],
                    [-0.2000,  0.2000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000],
                    [-0.2000, -0.2000,  0.0000],
                    [-0.2000, -0.2000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000],
                    [ 0.2000, -0.2000,  0.0000],
                    [ 0.2000, -0.2000,  0.0000]])
        )
        >>> t.node_names
        ['torso', 'front_left_leg', 'aux_1', 'front_left_foot', 'front_right_leg', 'aux_2', 'front_right_foot', 'left_back_leg', 'aux_3', 'left_back_foot', 'right_back_leg', 'aux_4', 'right_back_foot']
        >>> t.parent_indices
        tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  0, 10, 11])
        >>> t.local_translation
        tensor([[ 0.0000,  0.0000,  0.7500],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.2000,  0.2000,  0.0000],
                [ 0.2000,  0.2000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [-0.2000,  0.2000,  0.0000],
                [-0.2000,  0.2000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [-0.2000, -0.2000,  0.0000],
                [-0.2000, -0.2000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.2000, -0.2000,  0.0000],
                [ 0.2000, -0.2000,  0.0000]])
        >>> t.parent_of('front_left_leg')
        'torso'
        >>> t.index('front_right_foot')
        6
        >>> t[2]
        'aux_1'
    """

    __example_mjcf_path__ = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "tests/ant.xml"
    )

    def __init__(self, node_names, parent_indices, local_translation):
        """
        :param node_names: a list of names for each tree node
        :type node_names: list[str]
        :param parent_indices: an int32-typed tensor that represents the edge to its parent.        -1 represents the root node
        :type parent_indices: Tensor
        :param local_translation: a 3d vector that gives local translation information
        :type local_translation: Tensor
        """
        ln, lp, ll = (len(node_names), len(parent_indices), len(local_translation))
        assert len({ln, lp, ll}) == 1
        self._node_names = node_names
        self._parent_indices = parent_indices.long()
        self._local_translation = local_translation
        self._node_indices = {self.node_names[i]: i for i in range(len(self))}

    def __len__(self):
        """number of nodes in the skeleton tree"""
        return len(self.node_names)

    def __iter__(self):
        """iterator that iterate through the name of each node"""
        yield from self.node_names

    def __getitem__(self, item):
        """get the name of the node given the index"""
        return self.node_names[item]

    def __repr__(self):
        return f"SkeletonTree(\n    node_names={self._indent(repr(self.node_names))},\n    parent_indices={self._indent(repr(self.parent_indices))},\n    local_translation={self._indent(repr(self.local_translation))}\n)"

    def _indent(self, s):
        return "\n    ".join(s.split("\n"))

    @property
    def node_names(self):
        return self._node_names

    @property
    def parent_indices(self):
        return self._parent_indices

    @property
    def local_translation(self):
        return self._local_translation

    @property
    def num_joints(self):
        """number of nodes in the skeleton tree"""
        return len(self)

    @classmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        return cls(
            list(map(str, dict_repr["node_names"])),
            TensorUtils.from_dict(dict_repr["parent_indices"], *args, **kwargs),
            TensorUtils.from_dict(dict_repr["local_translation"], *args, **kwargs),
        )

    def to_dict(self):
        return OrderedDict(
            [
                ("node_names", self.node_names),
                ("parent_indices", tensor_to_dict(self.parent_indices)),
                ("local_translation", tensor_to_dict(self.local_translation)),
            ]
        )

    @classmethod
    def from_mjcf(cls, path: str) -> "SkeletonTree":
        """
        Parses a mujoco xml scene description file and returns a Skeleton Tree.
        We use the model attribute at the root as the name of the tree.

        :param path:
        :type path: string
        :return: The skeleton tree constructed from the mjcf file
        :rtype: SkeletonTree
        """
        tree = ET.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        node_names = []
        parent_indices = []
        local_translation = []

        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            pos = np.fromstring(
                xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" "
            )
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            curr_index = node_index
            node_index += 1
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        return cls(
            node_names,
            torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            torch.from_numpy(np.array(local_translation, dtype=np.float32)),
        )

    def parent_of(self, node_name):
        """get the name of the parent of the given node

        :param node_name: the name of the node
        :type node_name: string
        :rtype: string
        """
        return self[int(self.parent_indices[self.index(node_name)].item())]

    def index(self, node_name):
        """get the index of the node

        :param node_name: the name of the node
        :type node_name: string
        :rtype: int
        """
        return self._node_indices[node_name]

    def drop_nodes_by_names(
        self, node_names: list[str], pairwise_translation=None
    ) -> "SkeletonTree":
        new_length = len(self) - len(node_names)
        new_node_names = []
        new_local_translation = torch.zeros(
            new_length, 3, dtype=self.local_translation.dtype
        )
        new_parent_indices = torch.zeros(new_length, dtype=self.parent_indices.dtype)
        parent_indices = self.parent_indices.numpy()
        new_node_indices: dict = {}
        new_node_index = 0
        for node_index in range(len(self)):
            if self[node_index] in node_names:
                continue
            tb_node_index = parent_indices[node_index]
            if tb_node_index != -1:
                local_translation = self.local_translation[node_index, :]
                while tb_node_index != -1 and self[tb_node_index] in node_names:
                    local_translation += self.local_translation[tb_node_index, :]
                    tb_node_index = parent_indices[tb_node_index]
                assert tb_node_index != -1, "the root node cannot be dropped"
                if pairwise_translation is not None:
                    local_translation = pairwise_translation[
                        tb_node_index, node_index, :
                    ]
            else:
                local_translation = self.local_translation[node_index, :]
            new_node_names.append(self[node_index])
            new_local_translation[new_node_index, :] = local_translation
            if tb_node_index == -1:
                new_parent_indices[new_node_index] = -1
            else:
                new_parent_indices[new_node_index] = new_node_indices[
                    self[tb_node_index]
                ]
            new_node_indices[self[node_index]] = new_node_index
            new_node_index += 1
        return SkeletonTree(new_node_names, new_parent_indices, new_local_translation)

    def keep_nodes_by_names(
        self, node_names: list[str], pairwise_translation=None
    ) -> "SkeletonTree":
        nodes_to_drop = list(filter(lambda x: x not in node_names, self))
        return self.drop_nodes_by_names(nodes_to_drop, pairwise_translation)
