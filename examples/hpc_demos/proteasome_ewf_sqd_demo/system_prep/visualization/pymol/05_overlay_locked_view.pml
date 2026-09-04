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
set dash_labels, 0
set label_size, 18
set label_outline_color, white

load raw/5LF7.cif, canonical_5LF7
load intermediate/chemistry_map/5LF7_Y_chemistry_shell_transplanted_BO2.pdb, bort_transplanted

hide everything, all

# Core receptor environment in 5LF7 auth numbering:
# biological Thr1 -> auth Thr2
select catalytic_thr, canonical_5LF7 and chain Y and resi 2
select pocket_wall_1, canonical_5LF7 and chain Y and resi 21+22   # Ala20, Thr21
select pocket_wall_2, canonical_5LF7 and chain Y and resi 48+50   # Gly47, Ala49
select neighbor_asp, canonical_5LF7 and chain Z and resi 125
select conserved_waters, canonical_5LF7 and resn HOH and ((chain Y and resi 437+462) or (chain Z and resi 406))

show sticks, catalytic_thr or pocket_wall_1 or pocket_wall_2 or neighbor_asp
show spheres, conserved_waters

color yellow, catalytic_thr
color blue, pocket_wall_1 or pocket_wall_2
color red, neighbor_asp
color cyan, conserved_waters

# restore heteroatom colors where useful
color oxygen, (catalytic_thr or pocket_wall_1 or pocket_wall_2 or neighbor_asp) and elem O
color nitrogen, (catalytic_thr or pocket_wall_1 or pocket_wall_2 or neighbor_asp) and elem N

label canonical_5LF7 and chain Y and resi 2 and name CA, "Thr1 (auth Thr2)"
label canonical_5LF7 and chain Y and resi 21 and name CA, "Ala20"
label canonical_5LF7 and chain Y and resi 22 and name CA, "Thr21"
label canonical_5LF7 and chain Y and resi 48 and name CA, "Gly47"
label canonical_5LF7 and chain Y and resi 50 and name CA, "Ala49"
label canonical_5LF7 and chain Z and resi 125 and name CA, "Asp125 (chain Z)"
label canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, "W437"
label canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, "W406"
label canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, "W462"

# Matched overlay of both ligands in the same fixed receptor frame
select ixazomib, canonical_5LF7 and resn 6V8 and chain Y and resi 306
select bortezomib, bort_transplanted and resn BO2
show sticks, ixazomib or bortezomib
color magenta, ixazomib
color orange, bortezomib
color oxygen, (ixazomib or bortezomib) and elem O
color nitrogen, (ixazomib or bortezomib) and elem N
color boron, (ixazomib or bortezomib) and elem B

distance thrB_ix, canonical_5LF7 and chain Y and resi 2 and name OG1, ixazomib and name B26
distance thrB_bo, canonical_5LF7 and chain Y and resi 2 and name OG1, bortezomib and name B26
distance w437_ix, canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, ixazomib and name O27
distance w437_bo, canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, bortezomib and name O28
distance w406_ix, canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, ixazomib and name O8
distance w406_bo, canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, bortezomib and name O8
distance w462_ix, canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, ixazomib and name O28
distance w462_bo, canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, bortezomib and name O27
color cyan, w437_ix or w437_bo or w406_ix or w406_bo or w462_ix or w462_bo
color magenta, thrB_ix
color orange, thrB_bo

orient ixazomib or bortezomib or catalytic_thr or pocket_wall_1 or pocket_wall_2 or neighbor_asp or conserved_waters
set_view (    -0.005989226,   -0.580318809,    0.814367354,    -0.625122547,   -0.633463681,   -0.456011593,     0.780508101,   -0.511806190,   -0.358977079,    -0.000467354,    0.001321052,  -87.609725952,    49.666439056,  184.219207764,   65.937095642,    69.072868347,  106.148338318,   20.000000000 )

# Optional output:
# ray 1800, 1200
# png visualization/renders/05_overlay_locked_view.png, dpi=300
