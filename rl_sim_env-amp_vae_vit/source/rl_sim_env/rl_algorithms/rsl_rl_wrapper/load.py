import onnx
import onnxruntime as ort


def verify_onnx_model(model_path, name):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"{name} ONNX validation passed!")
    except onnx.checker.ValidationError as e:
        print(f"{name} ONNX validation failed: {e}")


def load_onnx_model(model_path):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    return session


def onnx_run_inference(session, actor_obs, vae_obs):
    inputs = {
        "actor_obs": actor_obs,
        "vae_obs": vae_obs,
    }
    outputs = session.run(None, inputs)

    return outputs
