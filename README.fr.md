# RunPod ComfyUI Pod — MiniMax H3 / RTX 5090

**[English](README.md)** | **Français**

![CUDA](https://img.shields.io/badge/CUDA-13.3%20%7C%2012.9-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-red) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.30.0-purple) ![Model](https://img.shields.io/badge/MiniMax-H3-orange)

Pod RunPod persistant avec **ComfyUI** + **VSCode (code-server)**, construit
autour de la génération vidéo **MiniMax H3** sur **RTX 5090** (Blackwell, sm_120).

MiniMax H3 génère des clips jusqu'en **2K, 24 fps, 4–15 s avec audio stéréo natif**
(dialogue, effets et ambiance produits dans la même passe) à partir de texte,
d'images, de vidéo ou de références audio.

## Ce qui fait tenir ce pod

Les poids exploitables de H3 pèsent **42,5 GB** (T2V/I2V) à **63,4 GB** (avec R2V)
face à 32 GB de VRAM. Trois éléments comblent l'écart, tous nouveaux depuis
ComfyUI 0.27–0.30 :

| Brique | Rôle |
|---|---|
| **comfy-aimdo** | Allocateur VRAM dynamique. Charge les poids à la demande et les décharge sous pression. Actif par défaut sur NVIDIA. |
| **comfy-kitchen** | Bibliothèque de kernels ComfyUI : NVFP4, int8-convrot, AWQ w4a16, RoPE/AdaLN fusionnés. |
| **Poids pruned int8-convrot** | ~40 % des paramètres (modulation) remplacés par une table de lookup, le reste quantifié en int8. |

Le text encoder est livré en **NVFP4**, qui exige les tensor cores Blackwell —
d'où la cible RTX 5090 plutôt qu'une carte moins chère.

Les deux packages sont livrés en **wheels précompilés** : cette image ne compile
plus rien. SageAttention, que la génération précédente de ce pod compilait
depuis les sources, a disparu — son upstream est figé depuis janvier 2026 et les
couches int8 de H3 ne sont de toute façon pas en FP16/BF16.

## Démarrage rapide

### 1. Déployer sur RunPod

Profil de pod visé — les défauts de ce dépôt sont réglés pour lui :

| | |
|---|---|
| GPU | RTX 5090, 32 GB VRAM |
| RAM hôte | **92 GB** |
| vCPU | 16 |
| Prix | 0,99 $/h |

Les 92 GB de RAM sont le chiffre déterminant : un workflow H3 a besoin de
~42,5 GB résidents, ça tient donc largement. C'est pour ça que `FAST_DISK` vaut
`false` par défaut (voir [RAM hôte](#la-vraie-contrainte-cest-la-ram-hôte)).

1. GPU : **RTX 5090** (32 GB VRAM / 92 GB RAM / 16 vCPU)
2. Container disk : 30 GB
3. **100 GB de stockage persistant sur `/workspace`** — les poids vivent là, pas
   dans l'image. Voir l'arbitrage ci-dessous.
4. Image : `vlop12ui/pod-comfyui-h3:latest`
5. Dans *Additional filters → CUDA Versions*, sélectionner **13.0+**

#### Volume disk ou network volume ?

Les deux se montent sur `/workspace` et cette image fonctionne avec l'un comme
l'autre. RunPod classe le volume disk en *fast (local)* et le network volume en
*variable (network)* :

| | Volume disk | Network volume |
|---|---|---|
| Vitesse | Local — chargement des poids plus rapide | Réseau — variable |
| Persistance | Jusqu'à **suppression** du pod | Indépendante du pod |
| Partageable entre pods | Non | Oui |
| Re-télécharger 63 GB à la suppression | Oui | Non |

Prendre le **volume disk** pour un pod unique et durable : charger 42 GB de
poids à chaque session est I/O-bound, le stockage local gagne. Prendre le
**network volume** si tu montes et démontes des pods, ou si plusieurs pods
partagent les mêmes poids.

Sur un **network volume**, mettre `PREWARM_SET=minimax-h3-fl2va`. Les lectures y
sont bornées par le réseau : sans ça, les 42 GB arrivent au fil de la première
génération, sous forme d'à-coups imprévisibles. Le préchargement transforme ça
en un coût unique et visible au démarrage. Ne nommer qu'un seul set —
précharger plus que ce qui tient en RAM ne fait que s'auto-évincer, et
`init.sh` saute l'étape si la RAM manque.

`FAST_DISK=true` n'a de sens que sur un volume disk, et encore, uniquement si la
RAM hôte manque.

**Dimensionnement** : 63,45 GB pour les deux variantes H3 laissent ~36 GB sur un
volume de 100 GB pour les outputs et inputs. Passer à 150 GB si tu génères
beaucoup ou si tu conserves les rushes.

> Si aucune machine CUDA 13 n'est disponible, utiliser `:cu129`. C'est le même
> PyTorch sur un runtime CUDA plus ancien, compatible hôtes 12.9.

### 2. Accès

- **ComfyUI** : `https://<pod-id>-3000.proxy.runpod.net`
- **VSCode** : `https://<pod-id>-8080.proxy.runpod.net`

Le premier démarrage télécharge 42 à 63 GB de poids. ComfyUI répond avant la fin
du téléchargement, mais H3 n'apparaîtra dans les loaders qu'une fois terminé.

### 3. Générer

ComfyUI 0.30 embarque les templates H3 : *Template Library → MiniMax H3 T2V / I2V / R2V*.

## Tags d'image

Chaque tag porte sa cible CUDA, les deux branches de build ne peuvent donc jamais
s'écraser mutuellement.

| Tag | Signification |
|---|---|
| `latest` | Dernier build `cu130` de `main` |
| `cu130` / `cu129` | Dernier build de cette cible |
| `cu130-main`, `cu129-develop` | Dernier build d'une cible sur une branche |
| `cu130-main-<sha>` | Commit exact — à utiliser pour la reproductibilité |
| `cu130-<date>-<sha>` | Triable chronologiquement |

## Stack

| Composant | Version | Note |
|---|---|---|
| Image de base | `nvidia/cuda:13.3.1-cudnn-runtime-ubuntu24.04` | `-runtime`, pas `-devel` : rien n'est compilé |
| PyTorch | 2.13.0+cu130 | index stable recommandé par ComfyUI |
| Python | 3.13 | dans un venv sur `/opt/venv` |
| ComfyUI | v0.30.0 | depuis `Comfy-Org/ComfyUI` (le repo a changé d'organisation) |
| comfy-kitchen | épinglé par ComfyUI | kernels NVFP4 / int8-convrot |
| comfy-aimdo | épinglé par ComfyUI | offloader VRAM dynamique |
| code-server | 4.131.0 | VSCode dans le navigateur |
| Driver requis | 580+ (cu130), 575+ (cu129) | fourni par l'hôte RunPod |

Les custom nodes sont réduits à un jeu orienté vidéo : ComfyUI-Manager,
VideoHelperSuite, KJNodes, rgthree, Frame-Interpolation, cg-use-everywhere. Les
suites de l'ère image (WAS, Impact Pack, Comfyroll, RES4LYF…) ont été retirées :
lourdes et cassantes à chaque montée de version du cœur. Easy-Use a été retiré
pour une raison précise : il dépend de `clip_interrogator` 0.6.0, publié en mars
2023, pari risqué face à transformers 5.x.

## Organisation du volume

ComfyUI reste **dans l'image** et n'est jamais copié sur le volume. Seules les
données mutables persistent. Mettre à jour ComfyUI revient donc à tirer un tag
plus récent, au lieu d'être masqué en permanence par une copie périmée sur le
volume — ce que faisait la conception précédente.

```
/workspace/comfyui-data/
├── models/
│   ├── diffusion_models/   # minimax_h3_*_pruned_int8_convrot.safetensors
│   ├── text_encoders/      # qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── vae/                # VAE vidéo (fp16) + audio (fp32)
│   └── loras/ upscale_models/ checkpoints/
├── custom_nodes/           # tes propres nodes, survivent aux mises à jour d'image
├── input/  output/  user/
└── extra_model_paths.yaml  # régénéré à chaque démarrage
```

## Configuration

Tout passe par des variables d'environnement — voir [.env.example](.env.example).
Les principales :

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_SETS` | `minimax-h3-fl2va,minimax-h3-ref2va` | Quels sets de `models/manifest.json` télécharger |
| `DOWNLOAD_MODELS` | `true` | `false` pour démarrer sans récupérer les poids |
| `FAST_DISK` | `false` | Échange RAM hôte contre disque à l'offload — utile seulement sur un pod pauvre en RAM |
| `PREWARM_SET` | — | Précharge un set dans le page cache au démarrage. Recommandé sur network volume |
| `VRAM_HEADROOM` | — | GB gardés libres ; à augmenter en cas d'OOM en cours de sampling |
| `COMFYUI_EXTRA_ARGS` | — | Ajouté tel quel à la ligne de commande ComfyUI |
| `CACHE_LRU` | — | `--cache-lru N` : évite de ré-encoder un prompt inchangé |
| `FAST_MODE` | — | Features `--fast` de ComfyUI, ou `all` |
| `ASYNC_OFFLOAD_STREAMS` | `2` | Streams d'offload des poids |
| `ENABLE_SSH` | `false` | Démarre sshd sur le port 22 |
| `PUBLIC_KEY` | — | Clé publique SSH pour root. Préférable au mot de passe |
| `SSH_PASSWORD` | — | Mot de passe root pour SSH. Bascule aussi `PermitRootLogin`, qu'Ubuntu laisse sinon sur `prohibit-password` |

### SSH

Tu n'en as probablement pas besoin : code-server sur 8080 te donne déjà un
terminal, et RunPod fournit le sien par-dessus. Laisse `ENABLE_SSH` sur `false`
sauf si quelque chose réclame vraiment le port 22.

Si tu l'actives, renseigne `PUBLIC_KEY` ou `SSH_PASSWORD`. Sans l'un des deux,
sshd écoute mais aucune connexion ne peut aboutir — root a un mot de passe
verrouillé et l'image ne contient aucun `authorized_keys`.

Les clés d'hôte vivent sur le network volume, dans `$COMFYUI_DATA_DIR/ssh`,
générées au premier démarrage. Elles sont donc propres à ton déploiement et
stables d'un redémarrage à l'autre : l'empreinte que ton client a mémorisée
reste valable.

### Régler la vitesse de génération

Sur un run mesuré, 12 steps ont pris 43,26 s dont ~30,8 s de sampling — les
**~12,5 s restantes sont de l'overhead**, et le log en donne la cause : le text
encoder de 15 GB est re-stagé à **chaque** prompt, même quand le texte n'a pas
changé.

| Overhead | Sampling |
|---|---|
| `CACHE_LRU=10` — réutilise le conditionnement | `FAST_MODE=fp16_accumulation` |
| `PREWARM_SET` — I/O du premier chargement | `FAST_MODE=cublas_ops` |
| `ASYNC_OFFLOAD_STREAMS=4` — ~40 GB par run sur PCIe | `FAST_MODE=autotune` |

Laquelle des deux moitiés domine dépend entièrement du workflow : sur un run
léger l'overhead pesait 29 % du total, sur un lourd environ 11 %.

**Avant de régler quoi que ce soit, détermine si c'est seulement compute-bound.**
Un accélérateur tiers supprime 30-35 % des évaluations du transformer pour 2,6 %
de temps réel — ce qui désigne les ~40,5 GB qui traversent le PCIe à chaque run,
pas le calcul. Un run de banc tranche :

```bash
python /app/scripts/bench.py wf.json --config fp16 --config offload4 --config offload8
```

Si `offload*` bouge et pas `fp16`, `--fast` est le mauvais endroit où investir —
et le vrai levier devient une carte qui n'a pas besoin d'offloader du tout. Le
panorama complet et les impasses vérifiées sont dans [VERSIONS.md §11](VERSIONS.md).

### Banc de test

Comparer deux générations à l'œil ne prouve rien : le temps par step sur ce pod
a été mesuré à 0,85 s, 2,57 s et 8,07 s — un facteur 9,5 dû uniquement aux
réglages du workflow. [`scripts/bench.py`](scripts/bench.py) supprime cette
variance.

```bash
python /app/scripts/bench.py mon_workflow_api.json --config fp16 --config lru
python /app/scripts/bench.py mon_workflow_api.json --sweep steps=8,12,16,20
python /app/scripts/bench.py --list
```

Exporte le workflow avec **Export (API)** — le format de sauvegarde normal est
refusé par `/prompt`.

Il lance ses propres instances ComfyUI sur le port 3111 (les leviers sont des
arguments CLI, et le watchdog de `start.sh` relancerait ton instance avec ses
arguments d'origine), épingle tous les widgets `seed`, chronomètre depuis
l'historique ComfyUI plutôt qu'autour de l'appel HTTP, jette le premier run de
chaque configuration, et désactive custom nodes et previews.

**Lis toujours le `spread` avant de croire un delta** : s'il dépasse l'écart
entre deux configs, tu as mesuré du bruit.

### Résolution et steps : où est le point d'équilibre ?

Aucune courbe n'est publiée — le modèle a quelques jours. Mais une partie de la
question a une réponse structurelle.

**Le canvas natif de H3 est de 768 px de petit côté, plafonné à 768×1344,
arrondi au multiple de 32** — soit ~1,0 mégapixel en 16:9. C'est là que le
modèle a appris. Au-dessus, la doc amont est nette : les pixels supplémentaires
*« may add pixels without adding equivalent learned detail »*. Le 2K annoncé
vient d'une régénération interne, pas d'un canvas plus grand.

Le plafond qui vaut d'être payé est donc **1344×768**. Au-delà, génère en natif
puis passe un vrai upscaler — `4x-UltraSharp` est dans l'image.

Pour les steps, deux presets : **12** (vitesse) et **20** (qualité).

Et le budget qui gouverne réellement le coût est **pixels × frames**, pas la
résolution seule. C'est ce qui a produit 2,57 s et 8,07 s par step ici, avec un
staging de modèles identique.

`--sweep` réutilise **une seule** instance ComfyUI : changer un widget ne change
pas la ligne de commande, recharger 40 GB par point ne prouverait rien.

Le banc ne mesure que le **temps**. Le second axe — à partir de quand des steps
supplémentaires cessent de se voir — reste à ton œil : seed fixe, sorties côte
à côte.

### Ajouter un modèle

Ajouter une entrée dans [models/manifest.json](models/manifest.json) et la
référencer dans `MODEL_SETS`. Aucune modification de `init.sh` ni du
`Dockerfile` :

```json
"mon-modele": {
  "description": "...",
  "default": false,
  "files": [
    { "repo": "org/repo", "path": "diffusion_models/x.safetensors",
      "dest": "diffusion_models", "size_gb": 12.3 }
  ]
}
```

Les sets peuvent se recouvrir — les fichiers partagés ne sont téléchargés qu'une
fois. Le downloader ignore ce qui est déjà présent, relancer sur un volume
existant ne coûte donc rien.

## La vraie contrainte, c'est la RAM hôte

Sur une carte 32 GB, le goulot d'étranglement est la **RAM système**, pas la
VRAM : l'offloader y fait transiter les poids.

Le chiffre qui compte est **~42,5 GB** — un modèle de diffusion plus le text
encoder et les deux VAE. Pas les 63,4 GB qu'occupe sur disque une installation
`fl2va` + `ref2va` complète : ce sont des alternatives, jamais résidentes
ensemble.

| RAM hôte | Verdict |
|---|---|
| 92 GB (pod visé) | Confortable — les poids tiennent en RAM avec de la marge |
| 48–80 GB | Jouable, peu de marge |
| < 48 GB | Swap ou OOM à prévoir |

`init.sh` indique au démarrage dans quelle tranche tombe le pod.

**À propos de `FAST_DISK`** : il fait offloader vers le disque plutôt que la RAM
hôte. Ça ne paie qu'avec du NVMe *local* rapide. Sur RunPod les poids sont sur
un **network volume**, donc l'activer sur un pod 92 GB est une pessimisation
doublement — d'où le défaut à `false`. `init.sh` alerte si tu l'actives quand
même sur un hôte à grosse RAM.

Si tu satures quand même la mémoire hôte, dans l'ordre :

1. `COMFYUI_EXTRA_ARGS=--disable-pinned-memory`
2. `COMFYUI_EXTRA_ARGS=--disable-pinned-memory --cache-none`
3. `FAST_DISK=true` — dernier recours, on échange de la vitesse contre la survie
4. Baisser résolution et durée, puis ne changer qu'une variable à la fois

Ne **pas** ajouter `--lowvram` : il désactive la VRAM dynamique, qui est
justement le mécanisme rendant H3 viable ici.

## Build

Les builds tournent sur GitHub Actions — voir [BUILD.md](BUILD.md). Le build
local reste possible pour un smoke test via `docker compose up --build`, mais ce
n'est pas le chemin de publication.

## Dépannage

| Symptôme | Cause | Correctif |
|---|---|---|
| Templates H3 absents | ComfyUI < 0.30.0 | Tirer un tag d'image plus récent |
| Modèle absent du loader | Téléchargement incomplet | Vérifier le log du pod ; relancer avec `DOWNLOAD_MODELS=true` |
| Erreur CUDA / driver au démarrage | Image cu130 sur un hôte 12.x | Redéployer avec le filtre CUDA, ou utiliser `:cu129` |
| R2V échoue, T2V fonctionne | Mauvais modèle de diffusion sélectionné | Choisir `minimax_h3_ref2va_*` et ajouter `minimax-h3-ref2va` à `MODEL_SETS` |
| Vidéo générée sans audio | VAE audio non branché | Les deux décodages VAE doivent alimenter le node `CreateVideo` |
| Clip légèrement plus long que demandé | Alignement de grille H3 (17k+5) | Normal : 5 s → 124 frames ≈ 5,17 s à 24 fps |

## Licence

AGPL-3.0 (héritée de ComfyUI).

**Les poids MiniMax H3 sont un sujet distinct, et potentiellement restrictif.**
Ils sont publiés sous MiniMax Community License. Une
[discussion HuggingFace](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/11)
rapporte que cette licence n'accorderait **aucun droit aux utilisateurs de l'UE,
des États-Unis, du Royaume-Uni et de Corée du Sud** en raison d'un litige en
cours — et que c'est précisément ce qui empêche la publication de variantes
distillées accélérées.

Il s'agit d'un rapport d'utilisateur, non vérifié ici contre le texte de la
licence. **Lis-la toi-même avant tout usage commercial de sorties H3** — la
France est dans le périmètre concerné.
