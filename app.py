
import streamlit as st
import torch
import numpy as np
from PIL import Image

from siamese_model import SiameseCNN


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="SignVerify",
    page_icon="✍️",
    layout="wide"
)


# ==============================
# CONFIGURATION
# ==============================

IMAGE_SIZE = (105, 105)
MODEL_PATH = "siamese_model_v3.pth"
THRESHOLD_PATH = "best_threshold_v3.txt"

with open(THRESHOLD_PATH, "r") as file:
    THRESHOLD = float(file.read().strip())

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    model = SiameseCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()
    return model


model = load_model()


# ==============================
# PREPROCESS IMAGE
# ==============================

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE)

    image = np.array(image, dtype=np.float32)
    image = image / 255.0

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32)


# ==============================
# BASIC BLANK IMAGE CHECK
# ==============================


def has_visible_ink(image):
    gray = np.array(image.convert("L"), dtype=np.float32)

    # Check whether the image has sufficient variation
    # to contain visible strokes.
    intensity_range = (
        np.percentile(gray, 95) -
        np.percentile(gray, 5)
    )

    contrast = np.std(gray)

    if intensity_range < 15 or contrast < 5:
        return False

    return True



# ==============================
# FEATURE EXTRACTION
# ==============================

def get_feature(image):
    image_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        feature = model.forward_one(image_tensor)

    return feature


# ==============================
# SIGNATURE VERIFICATION
# ==============================

def verify_signature(reference_images, test_image):
    test_feature = get_feature(test_image)

    distances = []

    for reference_image in reference_images:
        reference_feature = get_feature(reference_image)

        distance = torch.nn.functional.pairwise_distance(
            reference_feature,
            test_feature
        ).item()

        distances.append(distance)

    minimum_distance = min(distances)

    if minimum_distance < THRESHOLD:
        result = "GENUINE"
    else:
        result = "FORGED"

    return distances, minimum_distance, result


# ==============================
# APPLICATION UI
# ==============================

st.title("✍️ SignVerify")

st.subheader(
    "AI-Powered Offline Signature Verification"
)

st.write(
    "Verify a signature using multiple genuine "
    "reference signatures with a Siamese Neural Network."
)

st.divider()


# ==============================
# REFERENCE SIGNATURES
# ==============================

st.subheader("1️⃣ Reference Signatures")

reference_files = st.file_uploader(
    "Upload 3–5 genuine reference signatures",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="references"
)

reference_images = []

if reference_files:

    if len(reference_files) > 5:
        st.warning("Please upload a maximum of 5 reference signatures.")

    else:
        st.write(
            f"Reference signatures uploaded: {len(reference_files)}"
        )

        columns = st.columns(len(reference_files))

        for i, file in enumerate(reference_files):
            image = Image.open(file).convert("RGB")
            reference_images.append(image)

            with columns[i]:
                st.image(
                    image,
                    caption=f"Reference {i + 1}",
                    width=150
                )

st.divider()


# ==============================
# TEST SIGNATURE
# ==============================

st.subheader("2️⃣ Test Signature")

test_file = st.file_uploader(
    "Upload the signature you want to verify",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="test"
)

test_image = None

if test_file is not None:
    test_image = Image.open(test_file).convert("RGB")

    st.image(
        test_image,
        caption="Test Signature",
        width=250
    )

st.divider()


# ==============================
# VERIFY BUTTON
# ==============================

if st.button(
    "🔍 Verify Signature",
    use_container_width=True
):

    if not reference_files:
        st.warning("Please upload reference signatures.")

    elif len(reference_files) < 3:
        st.warning("Please upload at least 3 reference signatures.")

    elif len(reference_files) > 5:
        st.warning("Please upload a maximum of 5 reference signatures.")

    elif test_image is None:
        st.warning("Please upload a test signature.")

    elif not has_visible_ink(test_image):
        st.warning(
            "No visible signature detected. "
            "Please upload a clearer image."
        )

    else:

        with st.spinner("Analyzing signatures..."):

            distances, minimum_distance, result = verify_signature(
                reference_images,
                test_image
            )

        # ==============================
        # RESULTS
        # ==============================

        st.subheader("3️⃣ Verification Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Minimum Distance",
                f"{minimum_distance:.4f}"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.2f}"
            )

        st.write("### Individual Reference Comparisons")

        for i, distance in enumerate(distances):
            st.write(
                f"Reference {i + 1}: **{distance:.4f}**"
            )

        if result == "GENUINE":
            st.success("✅ GENUINE SIGNATURE")
        else:
            st.error("❌ FORGED SIGNATURE")

        st.info(
            f"Final Decision: {result} | "
            f"Minimum Distance: {minimum_distance:.4f} | "
            f"Threshold: {THRESHOLD:.2f}"
        )
