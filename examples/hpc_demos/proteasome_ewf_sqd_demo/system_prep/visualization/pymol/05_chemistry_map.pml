# 05_chemistry_map.pml
# Proteasome Challenge
# Chemistry-first active-site visualization in the frozen 5LF7-Y frame.
#
# Run from the proteasome_challenge project root:
#   pymol visualization/pymol/05_chemistry_map.pml
#
# Scientific notes:
# - 5LF7-Y is the frozen canonical receptor.
# - Biological Thr1 is author-numbered Thr2 in 5LF7.
# - BO2 is the rigidly transplanted bortezomib pose.
# - Waters shown here are screening candidates; H-bond assignments are not yet made.

reinitialize
set retain_order, 1
bg_color white
set orthoscopic, on
set antialias, 2
set depth_cue, 0
set ray_opaque_background, off
set cartoon_transparency, 0.72
set stick_radius, 0.16
set sphere_scale, 0.30
set dash_width, 2.0
set dash_gap, 0.30
set label_size, 18
set label_outline_color, white

# Inputs
load raw/5LF7.cif, canonical_5LF7
load intermediate/chemistry_map/5LF7_Y_chemistry_shell_transplanted_BO2.pdb, bort_transplanted

hide everything, all

# Ligands
select ixazomib, canonical_5LF7 and resn 6V8 and chain Y and resi 306
select bortezomib, bort_transplanted and resn BO2

show sticks, ixazomib
show sticks, bortezomib
color magenta, ixazomib
color orange, bortezomib
color oxygen, (ixazomib or bortezomib) and elem O
color nitrogen, (ixazomib or bortezomib) and elem N
color boron, (ixazomib or bortezomib) and elem B

# Prioritized receptor environment
select catalytic_thr, canonical_5LF7 and chain Y and resi 2
select strong_shared, canonical_5LF7 and chain Y and resi 21+22+48+50
select neighbor_asp, canonical_5LF7 and chain Z and resi 125
select contextual, canonical_5LF7 and chain Y and resi 34+170

show sticks, catalytic_thr or strong_shared or neighbor_asp or contextual
color yellow, catalytic_thr
color marine, strong_shared
color red, neighbor_asp
color gray60, contextual
color oxygen, (catalytic_thr or strong_shared or neighbor_asp or contextual) and elem O
color nitrogen, (catalytic_thr or strong_shared or neighbor_asp or contextual) and elem N
color sulfur, (catalytic_thr or strong_shared or neighbor_asp or contextual) and elem S

# Conserved crystallographic water candidates
select conserved_waters, canonical_5LF7 and resn HOH and ((chain Y and resi 437+462) or (chain Z and resi 406))
show spheres, conserved_waters
color cyan, conserved_waters

# Incompatible/contextual water identified by screening
select water_518, canonical_5LF7 and resn HOH and chain Y and resi 518
show spheres, water_518
color gray70, water_518
set sphere_transparency, 0.55, water_518

# Catalytic Thr-O--B geometry
distance thrB_ixazomib, canonical_5LF7 and chain Y and resi 2 and name OG1, ixazomib and name B26
distance thrB_bortezomib, canonical_5LF7 and chain Y and resi 2 and name OG1, bortezomib and name B26
color magenta, thrB_ixazomib
color orange, thrB_bortezomib
hide labels, thrB_ixazomib or thrB_bortezomib

# Conserved water-network distance guides
# Heavy-atom geometry only: these are NOT yet H-bond assignments.
distance w437_ix, canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, ixazomib and name O27
distance w437_bo, canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, bortezomib and name O28
distance w406_ix, canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, ixazomib and name O8
distance w406_bo, canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, bortezomib and name O8
distance w462_ix, canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, ixazomib and name O28
distance w462_bo, canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, bortezomib and name O27
color cyan, w437_ix or w437_bo or w406_ix or w406_bo or w462_ix or w462_bo
hide labels, w437_ix or w437_bo or w406_ix or w406_bo or w462_ix or w462_bo

# Labels
label canonical_5LF7 and chain Y and resi 2 and name CA, "Thr1 (auth Thr2)"
label canonical_5LF7 and chain Y and resi 22 and name CA, "Thr21"
label canonical_5LF7 and chain Y and resi 48 and name CA, "Gly47"
label canonical_5LF7 and chain Y and resi 50 and name CA, "Ala49"
label canonical_5LF7 and chain Y and resi 21 and name CA, "Ala20"
label canonical_5LF7 and chain Z and resi 125 and name CA, "Asp125 (chain Z)"
label canonical_5LF7 and chain Y and resi 34 and name CA, "Lys33"
label canonical_5LF7 and chain Y and resi 170 and name CA, "Tyr169"
label canonical_5LF7 and resn HOH and chain Y and resi 437 and name O, "W437"
label canonical_5LF7 and resn HOH and chain Z and resi 406 and name O, "W406"
label canonical_5LF7 and resn HOH and chain Y and resi 462 and name O, "W462"
label canonical_5LF7 and resn HOH and chain Y and resi 518 and name O, "W518?"

# Focus and viewing
select chemistry_focus, ixazomib or bortezomib or catalytic_thr or strong_shared or neighbor_asp or contextual or conserved_waters
orient chemistry_focus
zoom chemistry_focus, 5

# Useful interactive toggles:
#   disable contextual
#   disable water_518
#   disable bortezomib
#   disable ixazomib
#
# For a clean screenshot after choosing a view:
#   mkdir -p visualization/renders
#   ray 1800, 1200
#   png visualization/renders/05_chemistry_map.png, dpi=300
