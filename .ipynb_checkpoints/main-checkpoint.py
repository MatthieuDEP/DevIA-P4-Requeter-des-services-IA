import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm.auto import tqdm
import base64
import io
from dotenv import load_dotenv
import time


load_dotenv()

# TODO: Modifiez ces valeurs selon votre configuration
image_dir = "./assets/original"  # Exemple : si vous êtes sur Colab et avez uploadé un dossier
max_images = 50  # Commençons avec peu d'images

# IMPORTANT: Remplacez "VOTRE_TOKEN_HUGGING_FACE_ICI" par votre véritable token API.
# Ne partagez jamais votre token publiquement.
api_token = os.environ['HUGGING_TOKEN']

# Créons le dossier d'images s'il n'existe pas (pour l'exemple)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Dossier '{image_dir}' créé. Veuillez y ajouter des images .jpg ou .png.")
else:
    print(f"Dossier '{image_dir}' existant.")

if api_token != os.environ['HUGGING_TOKEN']:
    print("\nATTENTION : Vous devez remplacer 'VOTRE_TOKEN_HUGGING_FACE_ICI' par votre token API personnel.")



# Test requête
API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes" # Remplacez ... par le bon endpoint.
headers = {
    "Authorization": f"Bearer {api_token}"
    # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
}

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:max_images] # A vous de jouer !

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter : {image_paths}")  



# Fonctions utilitaires pour le traitement des masques
CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)

def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0 # Class ID for Background is 0

    return combined_mask



# Segmentation d'une seule image
if image_paths:
    single_image_path = image_paths[0] # Prenons la première image de notre liste
    print(f"Traitement de l'image : {single_image_path}")

    try:
        # Lire l'image en binaire
        # Et mettez le contenu de l'image dans la variable image_data
        with open(single_image_path, "rb") as f:
            image_data = f.read() # A vous de jouer !

        # Maintenant, utilisé l'API huggingface
        # ainsi que les fonctions données plus haut pour ségmenter vos images.
        headers_local = headers.copy()
        headers_local["Content-Type"] = "image/png"

        response = requests.post(
            API_URL, 
            headers=headers_local, 
            data=image_data
            )
        response.raise_for_status()
        results = response.json()

        width, height = get_image_dimensions(single_image_path)
        segmentation_mask = create_masks(results, width, height)
        print("Segmentation réussie, affichage des résultats...")

        original_image = Image.open(single_image_path)

        plt.figure(figsize=(10, 5))

        # Image originale
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Image originale")
        plt.axis("off")

        # Masque segmenté
        plt.subplot(1, 2, 2)
        plt.imshow(segmentation_mask, cmap="tab20")
        plt.title("Masque segmenté")
        plt.axis("off")

        plt.tight_layout()
        output_path = "segmentation_result.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Résultat sauvegardé dans : {output_path}")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
else:
    print("Aucune image à traiter. Vérifiez la configuration de 'image_dir' et 'max_images'.")



# Segmentation de plusieurs images
def segment_images_batch(list_of_image_paths):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.

    Args:
        list_of_image_paths (list): Liste des chemins vers les images.

    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
              Contient None si une image n'a pas pu être traitée.
    """
    batch_segmentations = []

    # Utilisation de tqdm pour afficher une barre de progression
    for img_path in tqdm(list_of_image_paths, desc="Segmentation des images", unit="image"):
        try:
            # Lecture de l'image
            with open(img_path, "rb") as f:
                image_data = f.read()

            # Appel API
            width, height = get_image_dimensions(img_path)

            headers_local = headers.copy()
            headers_local["Content-Type"] = "image/png"

            response = requests.post(
                API_URL,
                headers=headers_local,
                data=image_data
            )
            response.raise_for_status()
            results = response.json()

            # Création du masque
            mask = create_masks(results, width, height)
            batch_segmentations.append(mask)

            # Pause entre chaque appel API
            time.sleep(1)

        except Exception as e:
            print(f"Erreur sur {img_path} : {e}")
            batch_segmentations.append(None)

    return batch_segmentations

# Appeler la fonction pour segmenter les images listées dans image_paths
if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
    batch_seg_results = segment_images_batch(image_paths)
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")



# Affichage des résultats en Batch
CUSTOM_COLORMAP = {
    1: (255, 255, 0),     # Chapeau
    2: (255, 165, 0),     # Cheveux
    3: (255, 0, 255),     # Lunettes de soleil
    4: (255, 0, 0),       # Haut (vetement)
    5: (0, 255, 255),     # Jupe
    6: (0, 255, 0),       # Pantalon
    7: (0, 0, 255),       # Robe
    8: (128, 0, 128),     # Ceinture
    9: (255, 255, 0),     # Chaussure gauche
    10: (255, 140, 0),    # Chaussure droite
    11: (200, 180, 140),  # Visage
    12: (200, 180, 140),  # Jambe gauche
    13: (200, 180, 140),  # Jambe droite
    14: (200, 180, 140),  # Bras gauche
    15: (200, 180, 140),  # Bras droit
    16: (0, 128, 255),    # Sac
    17: (255, 20, 147)    # Echarpe
}

LEGEND_LABELS = {
    1: "Chapeau",
    2: "Cheveux",
    3: "Lunettes de soleil",
    4: "Haut (vetement)",
    5: "Jupe",
    6: "Pantalon",
    7: "Robe",
    8: "Ceinture",
    9: "Chaussure gauche",
    10: "Chaussure droite",
    11: "Visage",
    12: "Jambe gauche",
    13: "Jambe droite",
    14: "Bras gauche",
    15: "Bras droit",
    16: "Sac",
    17: "Echarpe"
}

def colorize_mask(mask, colormap):
    """Convertit un masque de labels en image RGB colorisée."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in colormap.items():
        colored[mask == label] = color

    return colored


def overlay_image_mask(image, colored_mask, alpha=0.3):
    """Superpose le masque colorisé à l'image originale."""
    image_np = np.array(image)
    overlay = (image_np * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    return overlay

def create_legend_patches_from_mask(mask, colormap, labels):
    """Crée une légende dynamique basée sur les classes présentes dans le masque."""
    patches = []

    present_classes = np.unique(mask)
    present_classes = present_classes[present_classes != 0]  # retirer background

    for class_id in present_classes:
        if class_id not in colormap:
            continue
        color = np.array(colormap[class_id]) / 255.0
        name = labels.get(class_id, f"Classe {class_id}")
        patches.append(mpatches.Patch(color=color, label=name))

    return patches


def display_segmented_images_batch(original_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés.

    Args:
        original_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
    """
    # Matplotlib, ça vous parle ?
    # Alors... au travail ! 😉

    os.makedirs("assets/segmented", exist_ok=True)

    for img_path, mask in zip(original_image_paths, segmentation_masks):
        if mask is None:
            continue

        image = Image.open(img_path)

        legend_patches = create_legend_patches_from_mask(
            mask, CUSTOM_COLORMAP, LEGEND_LABELS
        )

        colored_mask = colorize_mask(mask, CUSTOM_COLORMAP)
        overlay = overlay_image_mask(image, colored_mask)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(image)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(colored_mask)
        axes[1].set_title("Masque segmenté")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Superposition")
        axes[2].axis("off")

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=6,
            bbox_to_anchor=(0.5, -0.05)
        )

        filename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = f"assets/segmented/{filename}_segmented.png"

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Visualisation sauvegardée : {output_path}")

# Afficher les résultats du batch
if batch_seg_results:
    display_segmented_images_batch(image_paths, batch_seg_results)
else:
    print("Aucun résultat de segmentation à afficher.")