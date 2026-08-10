Input images baked into the image and seeded onto the volume by `start.sh`.

`--input-directory` points at the network volume, so ComfyUI's own bundled
`input/` is never visible and a fresh pod starts with an empty input
directory. Anything here is copied to `$DATA_DIR/input` on first boot and
never overwritten afterwards - an image you replaced is yours.

`transparent_rgb_gaming_mouse.png` is the file the official Comfy-Org H3
templates name. It is not published in their repository, so seeding it here
is what makes those templates - and the bench - runnable on any pod rather
than only on the machine a graph was exported from.
