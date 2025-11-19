
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.cm as cm
import time
from captum.attr import LayerGradCam 

# --- 1. Page Config ---
st.set_page_config(page_title="PaPr Final Demo", layout="wide")

# --- 2. Model Loading (Cached) ---
@st.cache_resource
def load_models():
    '''Load and cache the ScoreNet (MobileNetV2) and the Classifier (ResNet50)'''
    # ScoreNet (PaPr and Hybrid scoring)
    score_net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to('cpu').eval()
    # Classifier (Grad-CAM and Hybrid scoring - requires backward pass)
    classifier_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to('cpu').eval()
    
    return score_net, classifier_model

# --- 3. Configuration ---
IMG_SIZE = 224
PATCH_SIZE = 16
N_PATCHES = 14
N_TOTAL = N_PATCHES ** 2
DEVICE = 'cpu' 

# --- 4. Core Visualization Helpers ---
def create_overlay(heatmap_np, original_pil, alpha=0.6):
    '''Superimposes the heatmap (Fixed opacity 0.6)'''
    original_resized = original_pil.resize((IMG_SIZE, IMG_SIZE))
    heatmap_resized = Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(np.array(heatmap_resized) / 255.0)
    heatmap_pil = Image.fromarray((heatmap_colored[:, :, :3] * 255).astype(np.uint8))
    
    return Image.blend(original_resized, heatmap_pil, alpha=alpha)

def get_pruned_img(img_tensor, mask):
    '''Applies the mask to the image tensor and converts to displayable numpy.'''
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    mask_up = F.interpolate(mask_tensor, size=(IMG_SIZE, IMG_SIZE), mode='nearest')
    pruned_tensor = img_tensor * mask_up
    
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    pruned_img = inv_norm(pruned_tensor.squeeze()).permute(1, 2, 0).numpy()
    return np.clip(pruned_img, 0, 1)

# --- 5. Unified Algorithm Runner ---

# Helper function to get raw PaPr score grid
def calculate_papr_raw_score(score_net, img_tensor):
    feature_map = None
    def hook(m, i, o): nonlocal feature_map; feature_map = o
    handle = score_net.features[18].register_forward_hook(hook)
    with torch.no_grad(): _ = score_net(img_tensor)
    handle.remove()
    heatmap = F.interpolate(feature_map, size=(N_PATCHES, N_PATCHES), mode='bilinear', align_corners=False)
    scores_raw = torch.norm(heatmap, p=2, dim=1, keepdim=True).squeeze().numpy()
    return scores_raw

# Helper function to get raw Grad-CAM score grid
def calculate_gradcam_raw_score(classifier_model, img_tensor):
    lgc = LayerGradCam(classifier_model, classifier_model.layer4)
    with torch.enable_grad():
        outputs = classifier_model(img_tensor)
        target_class_id = torch.argmax(outputs, dim=1).item()
        grad_attr = lgc.attribute(img_tensor, target=target_class_id)
    scores_raw = F.interpolate(grad_attr.detach(), size=(N_PATCHES, N_PATCHES), mode='bilinear').squeeze().numpy()
    return np.maximum(scores_raw, 0) # Apply ReLU

def get_algorithm_results(method_name, score_net, classifier_model, image_pil, keep_ratio, img_tensor):
    
    start_time = time.time()
    
    # 1. Score Calculation (High Overhead Zone)
    if method_name == 'PaPr':
        scores_raw = calculate_papr_raw_score(score_net, img_tensor)
    elif method_name == 'Grad-CAM':
        scores_raw = calculate_gradcam_raw_score(classifier_model, img_tensor)
    elif method_name == 'Hybrid':
        # Calculate BOTH PaPr and Grad-CAM scores
        scores_papr = calculate_papr_raw_score(score_net, img_tensor)
        scores_grad = calculate_gradcam_raw_score(classifier_model, img_tensor)
        # Element-wise multiplication (The Hybrid Logic)
        scores_raw = scores_papr * scores_grad 
    elif method_name == 'Edge':
        edges = image_pil.convert('L').filter(ImageFilter.FIND_EDGES).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        scores_raw = np.array(edges) / 255.0
        scores_raw = scores_raw.reshape(N_PATCHES, PATCH_SIZE, N_PATCHES, PATCH_SIZE).mean(axis=(1, 3))
    elif method_name == 'Naive':
        hsv = np.array(image_pil.convert('HSV').resize((IMG_SIZE, IMG_SIZE)))
        scores_raw = hsv[:, :, 1] / 255.0
        scores_raw = scores_raw.reshape(N_PATCHES, PATCH_SIZE, N_PATCHES, PATCH_SIZE).mean(axis=(1, 3))

    # 2. Final Pruning and Display Logic
    if scores_raw.max() > scores_raw.min():
        scores_final = (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min() + 1e-8)
    else:
        scores_final = scores_raw
        
    n_keep = int(N_TOTAL * keep_ratio)
    flat = scores_final.flatten()
    flat_sorted = np.sort(flat)[::-1]
    threshold = flat_sorted[max(0, n_keep - 1)] if n_keep < N_TOTAL else -1.0
    mask = (scores_final >= threshold).astype(float)
    
    overlay = create_overlay(scores_final, image_pil)
    pruned = get_pruned_img(img_tensor, mask)
    
    elapsed = time.time() - start_time
    
    return {'overlay': overlay, 'pruned': pruned, 'time': elapsed}

# --- 6. Main Application UI ---
st.title("PaPr: 4-Method Comparison Demo")
st.markdown("Upload an image to see PaPr's semantic pruning against baselines and our Hybrid approach.")

# Load models 
score_net, classifier_model = load_models()

with st.sidebar:
    st.header("Controls")
    st.success("Model Ready!")
    st.divider()
    
    # Pruning ratio slider
    ratio_percent = st.slider("Keep Patches (%)", 10, 100, 50, 5)
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Preprocess image once for all algorithms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor_for_pruning = transform(img).unsqueeze(0)

    # --- Run All 4 Methods ---
    # Running Grad-CAM and Hybrid will cause noticeable lag. This is expected.
    with st.spinner("Processing all algorithms... (Grad-CAM and Hybrid may take a few seconds)"):
        papr_results = get_algorithm_results('PaPr', score_net, classifier_model, img, ratio_percent/100, img_tensor_for_pruning)
        hybrid_results = get_algorithm_results('Hybrid', score_net, classifier_model, img, ratio_percent/100, img_tensor_for_pruning)
        edge_results = get_algorithm_results('Edge', score_net, classifier_model, img, ratio_percent/100, img_tensor_for_pruning)
        naive_results = get_algorithm_results('Naive', score_net, classifier_model, img, ratio_percent/100, img_tensor_for_pruning)

    img_display = img.resize((IMG_SIZE, IMG_SIZE)) 

    # --- ROW 1: PaPr ---
    st.markdown("### Method 1: PaPr")
    st.caption("Understands objects regardless of color or texture using MobileNetV2 features.")
    col1, col2, col3 = st.columns(3)
    with col1: st.image(img_display, caption="Original", use_container_width=True)
    with col2: st.image(papr_results['overlay'], caption=f"Overlay ({papr_results['time']:.3f}s)", use_container_width=True)
    with col3: st.image(papr_results['pruned'], caption="Pruned Result", use_container_width=True)
    st.divider()
    
    # --- ROW 2: Hybrid ---
    st.markdown("### Method 2: Hybrid (PaPr + Grad-CAM)")
    st.caption("Combines PaPr's general object awareness with Grad-CAM's decision-specific focus for superior masks.")
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1: st.image(img_display, caption="Original", use_container_width=True)
    with col_h2: st.image(hybrid_results['overlay'], caption=f"Overlay ({hybrid_results['time']:.3f}s)", use_container_width=True)
    with col_h3: st.image(hybrid_results['pruned'], caption="Pruned Result", use_container_width=True)
    st.divider()

    # --- ROW 3: Edge Density ---
    st.markdown("### Method 3: Edge Density")
    st.caption("Keeps 'busy' areas with lots of lines. Good for detailed textures, bad for smooth objects.")
    col4, col5, col6 = st.columns(3)
    with col4: st.image(img_display, caption="Original", use_container_width=True)
    with col5: st.image(edge_results['overlay'], caption=f"Overlay ({edge_results['time']:.3f}s)", use_container_width=True)
    with col6: st.image(edge_results['pruned'], caption="Pruned Result", use_container_width=True)
    st.divider()
    
    # --- ROW 4: Naive Saturation ---
    st.markdown("### Method 4: Naive Saturation")
    st.caption("Keeps colorful areas. Fails on black/white objects and often selects irrelevant background.")
    col7, col8, col9 = st.columns(3)
    with col7: st.image(img_display, caption="Original", use_container_width=True)
    with col8: st.image(naive_results['overlay'], caption=f"Overlay ({naive_results['time']:.3f}s)", use_container_width=True)
    with col9: st.image(naive_results['pruned'], caption="Pruned Result", use_container_width=True)
    st.divider()
    
    # --- CONCLUSION SECTION ---
    st.subheader("Final Conclusion")
    st.info(
        '''
        **Why PaPr and Hybrid Matter:**
        
        As observed, **PaPr consistently selects semantically important regions** (the actual object) accurately and quickly. 
        It far outperforms basic methods like Edge Density (which struggles with smooth objects) and Naive Saturation (which is color-biased).
        
        Our **Hybrid method demonstrates superior fine-grained selection**, showcasing how combining PaPr's speed with Grad-CAM's precision 
        can recover the highest accuracy. While slower, it proves the potential for intelligent, high-accuracy pruning.
        '''
    )

else:
    st.info("⬅️ Please upload an image in the sidebar to start the comparison.")
