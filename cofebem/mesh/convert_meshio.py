import meshio

input_file = "./hertz_cube_v2.msh"


vtk_output = "hertz_cube.vtk"
xdmf_output = "hertz_cube.xdmf"

mesh = meshio.read(input_file)

# meshio.write(vtk_output, mesh)

meshio.write(xdmf_output, mesh)

print(f"Converted '{input_file}' to:")
# print(f"  → {vtk_output} (for ParaView)")
print(f"  → {xdmf_output} (for FEniCSx)")
