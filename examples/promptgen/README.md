# Prompt-generation experiment - the arms

The question: **does a local vision-language model write better H3 prompts than
we do by hand?** Two candidate tools exist and they are not obviously ordered,
so both run against the same control.

| Arm | File here | Where the prompt comes from |
|---|---|---|
| `control` | `control.txt` | written by hand |
| `writer` | - | the H3 Prompt Writer extension, from the ComfyUI Extensions menu |
| `raw` | `system-prompt-raw.md` | the same model, one system prompt, no extension |

## Running it

Put one `.txt` per arm in a directory on the volume - the file name becomes the
arm name - and render them all under identical conditions:

```
mkdir -p /workspace/comfyui-data/prompts
cp /app/examples/promptgen/control.txt /workspace/comfyui-data/prompts/
# add writer.txt and raw.txt from the two generators

bash /app/scripts/promptgen.sh start
# ... write the prompts ...
bash /app/scripts/promptgen.sh unload

python /app/scripts/prompt_ab.py /workspace/comfyui-data/prompts --reps 2
```

`--reps 2` runs every arm at seed 1234 and 1235. One seed is not enough to
separate a better prompt from a lucky roll, and two is the cheapest number that
starts to.

Point the graph at your own references rather than the bundled placeholder:

```
python /app/scripts/prompt_ab.py /workspace/comfyui-data/prompts \
    --set 137.image=my_character.png \
    --set 139.image=my_prop.png
```

## Judging it

Watch the videos before reading the manifest. The arm names are in the file
paths, so hiding them takes deliberate effort - and knowing which one you are
watching is exactly how a preference for the tool you just installed gets
written down as a measurement.

What to look at, in the order these usually break:

1. **Did it follow the reference?** The face, the object, the palette. This is
   where the extension arm should win if it wins at all - it reads the images;
   the raw arm only reads your description of them.
2. **Did anything enter the frame that you did not ask for?** Naming a thing
   invites it. A prompt that mentions what must not appear often produces it.
3. **Is the shot structure legal?** `[Shot 1]` carries no timestamp, later
   shots do. Wrong here and H3 quietly degrades rather than failing.
4. **Does the audio match?** `overall_soundscape: N/A` means silence, not
   "unspecified".

Only then open `run-*.json` in `output/promptgen/`, which holds the full text of
every prompt beside the file it produced.
