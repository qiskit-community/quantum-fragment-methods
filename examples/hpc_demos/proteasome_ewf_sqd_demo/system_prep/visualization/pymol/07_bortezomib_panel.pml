# Proteasome Challenge — matched chemistry panel
# PyMOL Open Source 3.1.0 compatible
# Run from the proteasome_challenge project root.

reinitialize
set retain_order, 1
set auto_show_selections, 0
bg_color white
set orthoscopic, on
set antialias, 2
set depth_cue, 0
set ray_opaque_background, off
set stick_radius, 0.16
set sphere_scale, 0.30
set dash_width, 2.0
set dash_gap, 0.30
set label_size, 18
set label_outline_color, white

load raw/5LF7.cif, canonical_5LF7
load intermediate/chemistry_map/5LF7_Y_chemistry_shell_transplanted_BO2.pdb, bort_transplanted

hide everything, all

# Receptor selections in 5LF7 author numbering.
# Biological Thr1 = author-numbered Thr2.
select catalytic_thr, canonical_5LF7 and chain Y and resi 2
select pocket_wall_1, canonical_5LF7 and chain Y and resi 21+22
select pocket_wall_2, canonical_5LF7 and chain Y and resi 48+50
select neighbor_asp, canonical_5LF7 and chain Z and resi 125
select conserved_waters, canonical_5LF7 and resn HOH and ((chain Y and resi 437+462) or (chain Z and resi 406))

show sticks, catalytic_thr
show sticks, pocket_wall_1
show sticks, pocket_wall_2
show sticks, neighbor_asp
show spheres, conserved_waters

color yellow, catalytic_thr
color blue, pocket_wall_1
color blue, pocket_wall_2
color red, neighbor_asp
color cyan, conserved_waters

color oxygen, catalytic_thr and elem O
color nitrogen, catalytic_thr and elem N
color oxygen, pocket_wall_1 and elem O
color nitrogen, pocket_wall_1 and elem N
color oxygen, pocket_wall_2 and elem O
color nitrogen, pocket_wall_2 and elem N
color oxygen, neighbor_asp and elem O
color nitrogen, neighbor_asp and elem N

label canonical_5LF7 and chain Y and resi 2 and name CA, "Thr1 (auth Thr2)"
label canonical_5LF7 and chain Y and resi 21 and name CA, "Ala20"
label canonical_5LF7 and chain Y and resi 22 and name CA, "Thr21"
label canonical_5LF7 and chain Y and resi 48 and name CA, "Gly47"
label canonical_5LF7 and chain Y and resi 50 and name CA, "Ala49"
label canonical_5LF7 and chain Z and resi 125 and name CA, "Asp125 (chain Z)"
label canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, "W437"
label canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, "W406"
label canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, "W462"

select bortezomib, bort_transplanted and resn BO2
show sticks, bortezomib
color orange, bortezomib
color oxygen, bortezomib and elem O
color nitrogen, bortezomib and elem N
color boron, bortezomib and elem B

distance thrB_bo, canonical_5LF7 and chain Y and resi 2 and name OG1, bortezomib and name B26
distance w437_bo, canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, bortezomib and name O28
distance w406_bo, canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, bortezomib and name O8
distance w462_bo, canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, bortezomib and name O27

color orange, thrB_bo
color cyan, w437_bo
color cyan, w406_bo
color cyan, w462_bo

hide labels, thrB_bo
hide labels, w437_bo
hide labels, w406_bo
hide labels, w462_bo

set_view (\
    -0.005989226,   -0.580318809,    0.814367354,\
    -0.625122547,   -0.633463681,   -0.456011593,\
     0.780508101,   -0.511806190,   -0.358977079,\
    -0.000467354,    0.001321052,  -87.609725952,\
    49.666439056,  184.219207764,   65.937095642,\
    69.072868347,  106.148338318,   20.000000000 )

deselect
