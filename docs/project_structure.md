# COFEBEM Project Structure

---

## Directory Structure


```plaintext
/cofebem-pythom
  /cofebem                
    /bem              
      boundary_elements.py
    /fem              
      fem.py
      form.py
      linear_form.py
      bilinear_form.py
      /wrapers
        fem_wraper.py
        fenics.py
        mfem.py
        zset.py
      /bcs
        bc.py
        dirichlet.py
        neumann.py
        robin.py
    /contact          
      normal_contact_solver.py
      friction_contact_solver.py
    /utils 
      /meshes
      /linear_algebra
      /optimization
        ccg.py
        nnls.py
  /docs               
    project_structure.md
    naming_conventions.md
  /tests              
    /unit_tests       
      test_boundary_elements.py
    /integration_tests
  /examples           
    fem_bem_example.py
  /build              
    CMakeLists.txt

