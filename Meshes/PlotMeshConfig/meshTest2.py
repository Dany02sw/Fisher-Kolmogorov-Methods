from dolfin import Point

from Utilities.EnumUtilities  import MeshType, MeshStructure

MESH_GRID_NAME    = "rectangle"

# Version 1
MESH_GRID_CONFIGS = [
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 20,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 30,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 45,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 70,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 100,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
]

# Version 2
MESH_GRID_CONFIGS = [
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 10,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 20,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 35,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 55,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 70,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.RECTANGLE,
        "N"         : 100,
        "P1"        : Point(0.0, 0.0),
        "P2"        : Point(3.0, 1.0),
        "structure" : MeshStructure.UNSTRUCTURED,
    },
]