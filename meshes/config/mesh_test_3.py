from utilities.enum_utilities  import MeshType, BrainSection

MESH_GRID_NAME    = "brain"

MESH_GRID_CONFIGS = [
    {
        "mesh_type"   : MeshType.BRAIN_2D,
        "brain_plane" : BrainSection.SAGITTAL,
    },
    {
        "mesh_type"   : MeshType.BRAIN_2D,
        "brain_plane" : BrainSection.CORONAL,
    },
    {
        "mesh_type"   : MeshType.BRAIN_2D,
        "brain_plane" : BrainSection.HORIZONTAL,
    },
]