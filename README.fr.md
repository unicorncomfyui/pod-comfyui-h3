# RunPod ComfyUI Pod - Série RTX 5090

**[English](README.md)** | **Français**

![RunPod ComfyUI RTX5000](https://img.shields.io/badge/RunPod-Pod-blue) ![CUDA](https://img.shields.io/badge/CUDA-12.8.1-green) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![ComfyUI](https://img.shields.io/badge/ComfyUI-36357bb-orange)

Pod RunPod persistant avec **ComfyUI** + **VSCode (code-server)** optimisé pour **RTX 5090** (architecture Blackwell).

## Pourquoi ce Pod ?

✅ **Indépendant** - Votre propre environnement persistant
✅ **Interface VSCode web** - IDE complet dans le navigateur
✅ **ComfyUI prêt** - Optimisé pour workflows WAN 2.2
✅ **SageAttention en cache** - Démarrage 10s vs 2-3min
✅ **Performances optimisées** - CUDA 12.8.1, tcmalloc, PyTorch nightly
✅ **Support Network Volume** - Stockage persistant

## Démarrage Rapide

### 1. Récupérer l'Image Pré-construite

```bash
docker pull <username>/pod-comfyui-vscode:latest
```

### 2. Déployer sur RunPod

1. Allez sur [RunPod Pods](https://www.runpod.io/console/pods)
2. Cliquez **+ Deploy**
3. Sélectionnez **GPU**: RTX 5090
4. **Container Image**: `<username>/pod-comfyui-vscode:latest`
5. **Container Disk**: 20GB minimum
6. **Expose Ports**: `8080, 3000`
7. **(Optionnel)** Attacher un Network Volume
8. Cliquez **Deploy**

### 3. Accéder à Votre Pod

Une fois déployé, RunPod fournit les URLs :

- **VSCode**: `https://your-pod-id-8080.proxy.runpod.net`
- **ComfyUI**: `https://your-pod-id-3000.proxy.runpod.net`

Aucune authentification requise - RunPod gère la sécurité.

## Stack Technique

| Composant | Version | Usage |
|-----------|---------|-------|
| **CUDA** | 12.8.1-cudnn | Runtime GPU |
| **Python** | 3.11 | Dernière version stable |
| **PyTorch** | Nightly cu128 | Support RTX 5090 (sm_120) |
| **ComfyUI** | Commit 36357bb | Version stable |
| **SageAttention** | Commit 68de379 | Attention quantifiée INT8/FP16 |
| **code-server** | 4.96.2 | VSCode dans le navigateur |
| **tcmalloc** | Latest | Optimisation mémoire |

## Fonctionnalités

### Optimisations ComfyUI

- ⚡ **Cache SageAttention**: démarrage ~10s (vs 2-3min sans cache)
- 🎯 **WAN 2.2 prêt**: Workflows text-to-video et image-to-video
- 🧠 **tcmalloc activé**: Gestion mémoire efficace
- 📦 **Support Network Volume**: Modèles et cache persistants

### Environnement de Développement

- 💻 **VSCode dans le navigateur**: IDE complet avec terminal
- 🔌 **Sans authentification**: Sécurisé par proxy RunPod
- 📂 **Accès à /workspace**: Éditer custom nodes, workflows, scripts
- 🐍 **Python 3.11 + PyTorch**: Prêt pour le développement

### Diagnostics Système

- 🔍 **Diagnostics automatiques au démarrage**: Info GPU (nom, driver, VRAM, compute capability), version CUDA driver, CPU/RAM/Disque
- ✅ **Vérification PyTorch CUDA**: Vérifie l'accessibilité GPU et compatibilité à l'initialisation
- 📋 **Inspection environnement**: Toutes les variables d'environnement NVIDIA/CUDA affichées
- 🛠️ **Version CUDA toolkit**: Vérification version CUDA du container
- 🐛 **Prêt pour débogage**: Info système complète pour diagnostiquer les problèmes de compatibilité

## Structure Network Volume

```
/workspace/  (monté depuis /runpod-volume)
├── ComfyUI/                    # Installation ComfyUI
│   ├── models/
│   │   ├── checkpoints/        # Vos modèles (.safetensors)
│   │   ├── loras/
│   │   ├── vae/
│   │   └── unet/
│   ├── custom_nodes/           # Nodes personnalisés
│   ├── output/                 # Images/vidéos générées
│   └── input/                  # Images source
├── sageattention_cache/        # Cache compilé SageAttention
│   ├── SageAttention/
│   └── .commit_hash
└── vos-projets/                # Vos projets de développement
```

## Performances

### Temps de Démarrage

- **Avec cache SageAttention** (Network Volume): ~10-15s
- **Sans cache** (premier démarrage): ~2-3min (compilation)
- **Validation cache**: Automatique via hash de commit

### Temps de Génération (RTX 5090)

| Workflow | Résolution | Images | Temps |
|----------|-----------|--------|-------|
| WAN 2.2 t2v | 720p | 61 | ~50-55s |
| WAN 2.2 i2v | 720p | 61 | ~50-55s |
| Génération image | 1080x1920 | 1 | ~10-12s |

## Utilisation

### Accéder aux Services

1. **VSCode**: Cliquez sur le lien port `8080` dans le dashboard RunPod
   - Éditer custom nodes dans `/workspace/ComfyUI/custom_nodes/`
   - Créer des workflows
   - Développement Python

2. **ComfyUI**: Cliquez sur le lien port `3000` dans le dashboard RunPod
   - Charger des workflows
   - Générer images/vidéos
   - Tester custom nodes

### Ajouter des Custom Nodes

Via terminal VSCode :

```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/votre-custom-node.git
cd votre-custom-node
pip install -r requirements.txt
```

Puis redémarrer ComfyUI (stop/start pod).

### Ajouter des Modèles

Upload via explorateur de fichiers VSCode ou terminal :

```bash
# Dans /workspace/ComfyUI/models/checkpoints/
# Uploader vos fichiers .safetensors
```

## Développement Local

```bash
git clone https://github.com/votreusername/pod-comfyui-vscode.git
cd pod-comfyui-vscode
docker-compose up --build
```

Accès :
- VSCode: http://localhost:8080
- ComfyUI: http://localhost:3000

## Troubleshooting

### Échec compilation SageAttention

```bash
# Dans terminal VSCode ou SSH
rm -rf /workspace/sageattention_cache
# Redémarrer le pod
```

### ComfyUI ne charge pas les modèles

Vérifier le répertoire modèles :

```bash
ls -la /workspace/ComfyUI/models/checkpoints/
```

S'assurer que les fichiers ont les bonnes permissions.

### Port non accessible

Vérifier dans le dashboard RunPod :
- Le pod est en cours d'exécution
- Les ports 8080 et 3000 sont exposés
- Cliquer sur le lien du port (pas l'IP directe)

## Estimation des Coûts

**RTX 5090** (~$0.90/heure) :
- Temps de développement : Facturé à l'heure
- Utilisation active recommandée : 4-8 heures/jour
- Coût : ~$3.60-$7.20/jour pour développement actif

**Astuce** : Arrêter le pod quand non utilisé pour économiser.

## Licence

AGPL-3.0 (hérité de ComfyUI)

---

**Développé pour RunPod Pods**
- Base : CUDA 12.8.1 + cuDNN + Ubuntu 24.04
- Python 3.11 + PyTorch nightly
- ComfyUI + VSCode
- Optimisé pour RTX 5090

*Dernière mise à jour : Décembre 2025*
