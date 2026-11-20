"""Public Streamlit inference website for pretrained MobileNetV2 Tea vs Coffee CNN."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'outputs/tea_coffee_model.keras'
st.set_page_config(page_title='Tea vs Coffee | CNN Classifier', page_icon='🍵', layout='wide')
st.title('🍵 Tea vs Coffee ☕')
st.caption('Image classification using an ImageNet-pretrained MobileNetV2 CNN, fine-tuned with a binary classification head.')
st.info('This classifier was trained on **tea and coffee grocery product packages**, not prepared drinks in cups.')

@st.cache_resource
def get_model():
    return tf.keras.models.load_model(MODEL, compile=False)


def predict(img):
    arr = np.asarray(img.convert('RGB').resize((160,160)), dtype=np.float32)[None,...]
    ptea = float(get_model().predict(arr, verbose=0)[0,0])
    return ('Tea' if ptea >= .5 else 'Coffee'), max(ptea, 1 - ptea)

if not MODEL.exists():
    st.error('Trained model missing. Run Train_in_Colab.ipynb, then deploy the ZIP it produces.')
    st.stop()

with st.sidebar:
    st.header('Model details')
    st.write('**Backbone:** MobileNetV2 (ImageNet pretrained)')
    st.write('**Classes:** Coffee (0), Tea (1)')
    st.write('**Images:** 100 train, 20 validation, 4 held-out tests')
    metricfile = ROOT / 'outputs/metrics.json'
    if metricfile.exists():
        m = json.loads(metricfile.read_text())
        st.metric('Validation accuracy', f"{m['validation_accuracy']:.1%}")
        st.write(f"Training epochs: {m['epochs_run']}")
    st.caption('The confidence is a model score, not a calibrated probability.')

st.header('Try an image')
file = st.file_uploader('Upload a tea or coffee package photo', type=['png','jpg','jpeg'])
if file is not None:
    try:
        photo = Image.open(file).convert('RGB')
        label, score = predict(photo)
        left, right = st.columns([1,1])
        with left:
            st.image(photo, caption='Your uploaded image', use_container_width=True)
        with right:
            st.subheader(f'Prediction: {label}')
            st.metric('Model score', f'{score:.1%}')
    except Exception as exc:
        st.error(f'Unable to classify this image: {exc}')

st.header('Four sample images')
st.write('Four held-out images (two tea, two coffee) that were not used for training or validation. Select any sample to run the classifier.')
sample_paths = sorted((ROOT / 'samples').glob('*.png')) + sorted((ROOT / 'samples').glob('*.jpg'))
if len(sample_paths) != 4:
    st.warning('Four sample images not found; run the training notebook to package them.')
else:
    cols = st.columns(4)
    for i, path in enumerate(sample_paths):
        im = Image.open(path).convert('RGB')
        with cols[i]:
            st.image(im, use_container_width=True)
            truth = 'Tea' if path.name.lower().startswith('tea_') else 'Coffee'
            st.caption(f'Sample {i+1} · Actual: {truth}')
            if st.button(f'Classify sample {i+1}', key=f'sample_{i}'):
                label, score = predict(im)
                st.success(f'{label} · {score:.1%}')

rfile = ROOT / 'outputs/four_predictions.json'
if rfile.exists():
    with st.expander('See four-image test results'):
        st.dataframe(json.loads(rfile.read_text()), use_container_width=True, hide_index=True)
st.divider()
st.caption('Dataset: Freiburg Groceries, via PerceptiLabs/Recognizing-Groceries. For educational use. Never trust model output as definitive.')
