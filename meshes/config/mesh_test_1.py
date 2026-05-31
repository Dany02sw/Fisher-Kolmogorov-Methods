from utilities.enum_utilities  import MeshType, MeshStructure

MESH_GRID_NAME    = "unit_square"

MESH_GRID_CONFIGS = [
    {
        "mesh_type" : MeshType.UNIT_SQUARE,
        "N"         : 4,
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.UNIT_SQUARE,
        "N"         : 8,
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.UNIT_SQUARE,
        "N"         : 16,
        "structure" : MeshStructure.UNSTRUCTURED,
    },
    {
        "mesh_type" : MeshType.UNIT_SQUARE,
        "N"         : 32,
        "structure" : MeshStructure.UNSTRUCTURED,
    },
]