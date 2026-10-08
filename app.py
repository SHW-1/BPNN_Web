import streamlit as st
import numpy as np
from scipy.io import loadmat
from pathlib import Path


# =========================================================
# BPNN-Based RAC Creep Prediction System
# MATLAB exported model:
# BPNN_Creep_Web_Model.mat
# =========================================================

MODEL_FILE = "BPNN_Creep_Web_Model.mat"
EXPECTED_INPUT_NUMBER = 8


# =========================================================
# 1. Activation functions
# =========================================================

def tansig(x):
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0


def logsig(x):
    return 1.0 / (1.0 + np.exp(-x))


def purelin(x):
    return x


def apply_activation(x, name):
    name = str(name).lower().strip()

    if name == "tansig":
        return tansig(x)
    elif name == "logsig":
        return logsig(x)
    elif name == "purelin":
        return purelin(x)
    else:
        raise ValueError(f"Unsupported activation function: {name}")


# =========================================================
# 2. Data conversion functions
# =========================================================

def to_column_vector(x):
    return np.asarray(x, dtype=float).reshape(-1, 1)


def to_1d_array(x):
    return np.asarray(x, dtype=float).reshape(-1)


def to_scalar(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty scalar value.")
    return float(arr[0])


def to_matrix(x, name):
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"{name} has unsupported shape: {arr.shape}")

    return arr


def matlab_string(x, default_value):
    if x is None:
        return default_value

    if isinstance(x, str):
        return x.strip()

    arr = np.asarray(x)

    if arr.dtype.kind in ["U", "S"]:
        return "".join(arr.reshape(-1).astype(str).tolist()).strip()

    try:
        return str(arr.item()).strip()
    except Exception:
        return default_value


def get_optional_vector(mat, name, length, default_value):
    if name not in mat:
        return np.full(length, default_value, dtype=float)

    arr = to_1d_array(mat[name])

    if arr.size != length:
        return np.full(length, default_value, dtype=float)

    return arr


# =========================================================
# 3. Load MATLAB exported model
# =========================================================

@st.cache_resource
def load_model():
    model_path = Path(MODEL_FILE)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. "
            f"Please put it in the same folder as app.py."
        )

    mat = loadmat(
        MODEL_FILE,
        squeeze_me=True,
        struct_as_record=False
    )

    required_vars = [
        "IW", "LW", "b1", "b2",
        "hidden_transferFcn", "output_transferFcn",
        "input_xoffset", "input_gain", "input_ymin",
        "output_xoffset", "output_gain", "output_ymin",
        "input_process_A", "input_process_b",
        "output_reverse_A", "output_reverse_b",
        "output_scale",
        "feature_default", "feature_min", "feature_max",
        "residual_low95", "residual_high95"
    ]

    for var in required_vars:
        if var not in mat:
            raise KeyError(f"Missing required variable in MAT file: {var}")

    W1 = to_matrix(mat["IW"], "IW")
    W2 = to_matrix(mat["LW"], "LW")

    input_process_A = to_matrix(mat["input_process_A"], "input_process_A")
    input_process_b = to_column_vector(mat["input_process_b"])

    output_reverse_A = to_matrix(mat["output_reverse_A"], "output_reverse_A")
    output_reverse_b = to_column_vector(mat["output_reverse_b"])

    b1 = to_column_vector(mat["b1"])
    b2 = to_column_vector(mat["b2"])

    input_xoffset = to_column_vector(mat["input_xoffset"])
    input_gain = to_column_vector(mat["input_gain"])
    input_ymin = to_scalar(mat["input_ymin"])

    output_xoffset = to_scalar(mat["output_xoffset"])
    output_gain = to_scalar(mat["output_gain"])
    output_ymin = to_scalar(mat["output_ymin"])

    output_scale = to_scalar(mat["output_scale"])

    if output_scale == 0:
        raise ValueError("output_scale cannot be zero.")

    hidden_transfer = matlab_string(
        mat["hidden_transferFcn"],
        "tansig"
    )

    output_transfer = matlab_string(
        mat["output_transferFcn"],
        "purelin"
    )

    residual_low95 = to_scalar(mat["residual_low95"])
    residual_high95 = to_scalar(mat["residual_high95"])

    feature_default = get_optional_vector(
        mat,
        "feature_default",
        EXPECTED_INPUT_NUMBER,
        0.0
    )

    feature_min = get_optional_vector(
        mat,
        "feature_min",
        EXPECTED_INPUT_NUMBER,
        np.nan
    )

    feature_max = get_optional_vector(
        mat,
        "feature_max",
        EXPECTED_INPUT_NUMBER,
        np.nan
    )

    # Dimension checks
    input_number = W1.shape[1]
    hidden_number = W1.shape[0]

    if input_number != input_process_A.shape[0]:
        raise ValueError(
            f"IW and input_process_A dimension mismatch. "
            f"IW shape: {W1.shape}, input_process_A shape: {input_process_A.shape}"
        )

    if input_process_A.shape[1] != EXPECTED_INPUT_NUMBER:
        raise ValueError(
            f"The model should receive {EXPECTED_INPUT_NUMBER} original inputs, "
            f"but input_process_A shape is {input_process_A.shape}"
        )

    if b1.shape[0] != hidden_number:
        raise ValueError(
            f"b1 size does not match hidden layer size. "
            f"Hidden size: {hidden_number}, b1 size: {b1.shape[0]}"
        )

    if W2.shape[1] != hidden_number:
        if W2.shape[0] == hidden_number and W2.shape[1] == 1:
            W2 = W2.T
        else:
            raise ValueError(
                f"LW shape does not match hidden layer size. "
                f"IW shape: {W1.shape}, LW shape: {W2.shape}"
            )

    if b2.shape[0] != W2.shape[0]:
        raise ValueError(
            f"b2 size does not match output layer size. "
            f"Output size: {W2.shape[0]}, b2 size: {b2.shape[0]}"
        )

    x_check = None
    y_check_web_matlab = None

    if "X_check" in mat:
        x_check = to_1d_array(mat["X_check"])

    if "y_check_web_matlab" in mat:
        y_check_web_matlab = to_scalar(mat["y_check_web_matlab"])

    return {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2,
        "hidden_transfer": hidden_transfer,
        "output_transfer": output_transfer,

        "input_xoffset": input_xoffset,
        "input_gain": input_gain,
        "input_ymin": input_ymin,

        "output_xoffset": output_xoffset,
        "output_gain": output_gain,
        "output_ymin": output_ymin,

        "input_process_A": input_process_A,
        "input_process_b": input_process_b,
        "output_reverse_A": output_reverse_A,
        "output_reverse_b": output_reverse_b,

        "output_scale": output_scale,
        "feature_default": feature_default,
        "feature_min": feature_min,
        "feature_max": feature_max,

        "residual_low95": residual_low95,
        "residual_high95": residual_high95,

        "x_check": x_check,
        "y_check_web_matlab": y_check_web_matlab,

        "input_number": EXPECTED_INPUT_NUMBER,
        "hidden_number": hidden_number,
    }


# =========================================================
# 4. MATLAB mapminmax
# =========================================================

def mapminmax_apply(x, model):
    """
    MATLAB mapminmax apply:
    y = (x - xoffset) * gain + ymin
    """
    return (
        (x - model["input_xoffset"])
        * model["input_gain"]
        + model["input_ymin"]
    )


def mapminmax_reverse(y_norm, model):
    """
    MATLAB mapminmax reverse:
    x = (y - ymin) / gain + xoffset
    """
    return (
        (y_norm - model["output_ymin"])
        / model["output_gain"]
        + model["output_xoffset"]
    )


# =========================================================
# 5. BPNN prediction
# =========================================================

def predict_creep(input_values, model):
    x = to_column_vector(input_values)

    # External input normalization using ps_input
    x_norm = mapminmax_apply(x, model)

    # MATLAB network internal input processing
    x_net = model["input_process_A"] @ x_norm + model["input_process_b"]

    # Hidden layer
    hidden_net = model["W1"] @ x_net + model["b1"]
    hidden_output = apply_activation(
        hidden_net,
        model["hidden_transfer"]
    )

    # Output layer
    output_net = model["W2"] @ hidden_output + model["b2"]
    output_internal = apply_activation(
        output_net,
        model["output_transfer"]
    )

    # MATLAB network internal output reverse processing
    y_norm = (
        model["output_reverse_A"] @ output_internal
        + model["output_reverse_b"]
    )

    # External output reverse normalization using ps_output
    y_raw = mapminmax_reverse(y_norm, model)

    # Web display scale
    y_web = y_raw / model["output_scale"]

    return float(y_web.reshape(-1)[0])


# =========================================================
# 6. Streamlit interface
# =========================================================

st.set_page_config(
    page_title="BPNN Creep Prediction System",
    page_icon="📈",
    layout="centered"
)

st.title("BPNN-Based RAC Creep Prediction System")

st.markdown(
    """
    This web application predicts the **creep of recycled aggregate concrete (RAC)**
    using a MATLAB-trained BPNN model.
    """
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


feature_labels = [
    "Ordinary Portland cement content (Y1)",
    "RAC sand ratio (Y2)",
    "RAC water-binder ratio (Y3)",
    "Mixed coarse aggregate water absorption (Y4)",
    "Creep test humidity (Y5)",
    "Compressive strength (Y6)",
    "Stress ratio (Y7)",
    "Creep test duration (Y8)"
]

output_label = "Predicted creep of RAC (Z)"

st.subheader("Input Parameters")

input_values = []

cols = st.columns(2)

for i, label in enumerate(feature_labels):
    default_value = float(model["feature_default"][i])
    min_value = model["feature_min"][i]
    max_value = model["feature_max"][i]

    help_text = None

    if not np.isnan(min_value) and not np.isnan(max_value):
        help_text = f"Training data range: {min_value:.6g} to {max_value:.6g}"

    if not np.isnan(min_value) and not np.isnan(max_value) and max_value > min_value:
        step_value = float((max_value - min_value) / 100.0)
    else:
        step_value = 0.01

    with cols[i % 2]:
        value = st.number_input(
            label=label,
            value=default_value,
            step=step_value,
            format="%.6f",
            help=help_text,
            key=f"input_{i}"
        )

        input_values.append(value)


st.markdown("---")

if st.button("Predict", use_container_width=True):
    try:
        prediction = predict_creep(input_values, model)

        lower95 = prediction + model["residual_low95"]
        upper95 = prediction + model["residual_high95"]

        st.success(f"{output_label}: {prediction:.6f}")

        st.markdown("### Uncertainty Analysis")

        st.info(
            f"""
            **95% Predictive Interval:** [{lower95:.6f}, {upper95:.6f}]

            **Interval method:** Residual calibration based on the test set.
            """
        )

        st.markdown("### Input Summary")

        for label, value in zip(feature_labels, input_values):
            st.write(f"{label}: {value:.6f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.markdown("---")

with st.expander("Model Consistency Check"):
    st.write(f"Model file: `{MODEL_FILE}`")
    st.write(f"Hidden layer activation function: `{model['hidden_transfer']}`")
    st.write(f"Output layer activation function: `{model['output_transfer']}`")
    st.write(f"Input number: `{model['input_number']}`")
    st.write(f"Hidden layer neurons: `{model['hidden_number']}`")
    st.write(f"Output scale: `{model['output_scale']}`")
    st.write(f"IW shape: `{model['W1'].shape}`")
    st.write(f"LW shape: `{model['W2'].shape}`")
    st.write(f"input_process_A shape: `{model['input_process_A'].shape}`")
    st.write(f"output_reverse_A shape: `{model['output_reverse_A'].shape}`")
    st.write(f"95% residual lower bound: `{model['residual_low95']}`")
    st.write(f"95% residual upper bound: `{model['residual_high95']}`")

    if model["x_check"] is not None and model["y_check_web_matlab"] is not None:
        py_check = predict_creep(model["x_check"], model)
        matlab_check = model["y_check_web_matlab"]
        diff = py_check - matlab_check

        st.write("MATLAB exported check input:")
        st.write(model["x_check"])

        st.write(f"MATLAB reference prediction: `{matlab_check:.10f}`")
        st.write(f"Python reproduced prediction: `{py_check:.10f}`")
        st.write(f"Difference: `{diff:.10e}`")
